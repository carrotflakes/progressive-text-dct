"""Model components: orthonormal DCT, frozen partial encoder, trainable decoder.

Decoder input layout (teacher forcing):
    [len_emb(n)] [proj(z_1)+idx_emb(i_1)] ... [proj(z_K)+idx_emb(i_K)] [bos] t_1 ... t_n
Labels are -100 everywhere except the token positions t_1..t_n (HF shifts
internally, so logits at [bos] predict t_1, logits at t_{n-1} predict t_n).
"""

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------- DCT

def dct_matrix(n, device=None, dtype=torch.float32):
    """Orthonormal DCT-II matrix C (n x n): (C @ x) == scipy dct(x, 2, norm='ortho').
    Inverse is C.T (orthonormal)."""
    i = torch.arange(n, dtype=torch.float64)
    k = i.unsqueeze(1)  # (n, 1)
    c = torch.cos(math.pi * (2 * i.unsqueeze(0) + 1) * k / (2 * n))
    scale = torch.full((n, 1), math.sqrt(2.0 / n), dtype=torch.float64)
    scale[0, 0] = math.sqrt(1.0 / n)
    return (scale * c).to(device=device, dtype=dtype)


class DCTBank:
    """Cache of DCT matrices per sequence length."""

    def __init__(self, device, dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.mats = {}

    def get(self, n):
        if n not in self.mats:
            self.mats[n] = dct_matrix(n, self.device, self.dtype)
        return self.mats[n]


# ---------------------------------------------------------------- encoder

class PartialEncoder(nn.Module):
    """Frozen encoder = embedding layer + first L transformer layers of the
    base LM (weight-shared with the decoder LM; run with adapters disabled).

    Right padding + causal attention means real positions never attend to
    padding, so no attention mask is needed for correctness.
    """

    def __init__(self, backbone, layer):
        super().__init__()
        self.backbone = backbone  # reference, not copy
        self.layer = layer
        self.partial_ok = None  # decided on first call by self-test

    @torch.no_grad()
    def _full(self, input_ids):
        out = self.backbone(input_ids=input_ids, output_hidden_states=True)
        return out.hidden_states[self.layer]

    @torch.no_grad()
    def _partial(self, input_ids):
        bb = self.backbone
        hidden = bb.embed_tokens(input_ids)
        if self.layer == 0:
            return hidden
        pos = torch.arange(input_ids.shape[1], device=input_ids.device)
        pos = pos.unsqueeze(0).expand(input_ids.shape[0], -1)
        kwargs = {"attention_mask": None, "position_ids": pos}
        if hasattr(bb, "rotary_emb"):
            kwargs["position_embeddings"] = bb.rotary_emb(hidden, pos)
        for lyr in bb.layers[: self.layer]:
            out = lyr(hidden, **kwargs)
            hidden = out[0] if isinstance(out, tuple) else out
        return hidden

    @torch.no_grad()
    def forward(self, input_ids):
        if self.partial_ok is None:
            try:
                h_p = self._partial(input_ids)
                h_f = self._full(input_ids)
                diff = (h_p.float() - h_f.float()).abs().max().item()
                scale = h_f.float().abs().mean().item() + 1e-6
                self.partial_ok = diff / scale < 1e-2
                print(f"[encoder] partial-forward self-test: maxdiff={diff:.2e} "
                      f"(rel {diff / scale:.2e}) -> "
                      f"{'using partial' if self.partial_ok else 'FALLBACK to full'}")
            except Exception as e:  # noqa: BLE001
                print(f"[encoder] partial forward failed ({e}); falling back to full")
                self.partial_ok = False
        return self._partial(input_ids) if self.partial_ok else self._full(input_ids)


# ---------------------------------------------------------------- decoder extras

class DecoderExtras(nn.Module):
    """Fully-trained (non-LoRA) decoder parameters."""

    def __init__(self, d_enc, d_model, n_max):
        super().__init__()
        self.proj = nn.Linear(d_enc, d_model)
        self.idx_emb = nn.Embedding(n_max, d_model)    # coefficient index 0..n_max-1
        self.len_emb = nn.Embedding(n_max + 1, d_model)  # original length n
        self.bos = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.idx_emb.weight, std=0.02)
        nn.init.normal_(self.len_emb.weight, std=0.02)
        nn.init.normal_(self.bos, std=0.02)


class Compressor:
    """Frozen compression: hidden states -> selected (DCT) coefficient rows."""

    def __init__(self, dct_bank, mode, h_scale):
        self.bank = dct_bank
        self.mode = mode  # "dct" | "trunc"
        self.h_scale = h_scale

    def compress(self, h, idx):
        """h: (n, d) hidden states of one sample (real positions only).
        idx: list of kept coefficient/token indices. Returns (k, d)."""
        h = h.float() / self.h_scale
        if self.mode == "trunc":
            return h[idx]
        c = self.bank.get(h.shape[0])
        return (c @ h)[idx]


def build_decoder_batch(items, hiddens, compressor, extras, embed_tokens,
                        device, pad_to=None):
    """Build right-padded teacher-forcing inputs.

    items: list of {tokens, n, k, idx}; hiddens: list of (n_i, d) tensors.
    Returns inputs_embeds (B, L, d_model), attention_mask, labels.
    """
    d_model = extras.proj.out_features
    lens = [it["k"] + 2 + it["n"] for it in items]
    L = pad_to or max(lens)
    B = len(items)
    p_dtype = extras.proj.weight.dtype
    inputs = torch.zeros(B, L, d_model, device=device, dtype=p_dtype)
    mask = torch.zeros(B, L, dtype=torch.long, device=device)
    labels = torch.full((B, L), -100, dtype=torch.long, device=device)
    for b, it in enumerate(items):
        n, k, idx = it["n"], it["k"], it["idx"]
        z = compressor.compress(hiddens[b], idx).to(p_dtype)  # (k, d_enc)
        idx_t = torch.tensor(idx, device=device)
        zp = extras.proj(z) + extras.idx_emb(idx_t)
        toks = torch.tensor(it["tokens"], device=device)
        tok_emb = embed_tokens(toks).to(p_dtype)
        seq = torch.cat([
            extras.len_emb(torch.tensor([n], device=device)),
            zp,
            extras.bos.unsqueeze(0),
            tok_emb,
        ], dim=0)
        inputs[b, : seq.shape[0]] = seq
        mask[b, : seq.shape[0]] = 1
        labels[b, k + 2 : k + 2 + n] = toks
    return inputs, mask, labels


def build_prefixes(items, hiddens, compressor, extras, device):
    """Left-padded generation prefixes [len][z...][bos].

    Returns inputs_embeds (B, P, d), attention_mask (B, P)."""
    d_model = extras.proj.out_features
    p_dtype = extras.proj.weight.dtype
    plens = [it["k"] + 2 for it in items]
    P = max(plens)
    B = len(items)
    inputs = torch.zeros(B, P, d_model, device=device, dtype=p_dtype)
    mask = torch.zeros(B, P, dtype=torch.long, device=device)
    for b, it in enumerate(items):
        n, idx = it["n"], it["idx"]
        z = compressor.compress(hiddens[b], idx).to(p_dtype)
        idx_t = torch.tensor(idx, device=device)
        zp = extras.proj(z) + extras.idx_emb(idx_t)
        seq = torch.cat([
            extras.len_emb(torch.tensor([n], device=device)),
            zp,
            extras.bos.unsqueeze(0),
        ], dim=0)
        inputs[b, P - seq.shape[0]:] = seq
        mask[b, P - seq.shape[0]:] = 1
    return inputs, mask


@torch.no_grad()
def greedy_generate(lm, embed_tokens, prefix_embeds, prefix_mask, gen_lens,
                    autocast_dtype=torch.bfloat16):
    """Greedy decode exactly gen_lens[b] tokens per sample (KV cache).
    Returns list of token-id lists."""
    device = prefix_embeds.device
    B = prefix_embeds.shape[0]
    max_steps = max(gen_lens)
    pos = (prefix_mask.cumsum(-1) - 1).clamp(min=0)
    with torch.autocast("cuda", dtype=autocast_dtype):
        out = lm(inputs_embeds=prefix_embeds, attention_mask=prefix_mask,
                 position_ids=pos, use_cache=True)
    past = out.past_key_values
    next_tok = out.logits[:, -1].float().argmax(-1)  # (B,)
    toks = [next_tok]
    mask = prefix_mask
    cur_pos = pos[:, -1] + 1
    for _ in range(max_steps - 1):
        emb = embed_tokens(next_tok).unsqueeze(1)
        mask = torch.cat([mask, torch.ones(B, 1, dtype=torch.long, device=device)], dim=1)
        with torch.autocast("cuda", dtype=autocast_dtype):
            out = lm(inputs_embeds=emb, attention_mask=mask,
                     position_ids=cur_pos.unsqueeze(1), use_cache=True,
                     past_key_values=past)
        past = out.past_key_values
        next_tok = out.logits[:, -1].float().argmax(-1)
        toks.append(next_tok)
        cur_pos = cur_pos + 1
    toks = torch.stack(toks, dim=1)  # (B, max_steps)
    return [toks[b, : gen_lens[b]].tolist() for b in range(B)]


# ---------------------------------------------------------------- assembly

def load_base(cfg, device):
    """Load tokenizer + base LM (fp32 weights, autocast at runtime)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = cfg["model"]["base_model"]
    try:
        tok = AutoTokenizer.from_pretrained(name)
        lm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    except Exception as e:  # noqa: BLE001
        print(f"failed to load {name} ({e}); falling back to "
              f"{cfg['model']['fallback_model']}")
        name = cfg["model"]["fallback_model"]
        tok = AutoTokenizer.from_pretrained(name)
        lm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    lm.to(device)
    lm.config.use_cache = False
    return tok, lm


def apply_lora(lm, cfg):
    from peft import LoraConfig, get_peft_model

    lcfg = cfg["lora"]
    peft_cfg = LoraConfig(
        r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
        target_modules=list(lcfg["target_modules"]), bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(lm, peft_cfg)


def get_backbone(lm):
    """Qwen2ForCausalLM/GPT2LMHeadModel -> inner backbone (Qwen2Model etc.)."""
    base = lm.get_base_model() if hasattr(lm, "get_base_model") else lm
    return base.model if hasattr(base, "model") else base.transformer


@torch.no_grad()
def compute_h_scale(encoder, chunks, device, batch_size=32, pad_id=0,
                    autocast_dtype=torch.bfloat16):
    """Global RMS of encoder hidden states over real positions (scalar)."""
    tot, cnt = 0.0, 0
    for s in range(0, len(chunks), batch_size):
        batch = chunks[s : s + batch_size]
        maxn = max(len(c) for c in batch)
        ids = torch.full((len(batch), maxn), pad_id, dtype=torch.long, device=device)
        for b, c in enumerate(batch):
            ids[b, : len(c)] = torch.tensor(c, device=device)
        with torch.autocast("cuda", dtype=autocast_dtype):
            h = encoder(ids)
        for b, c in enumerate(batch):
            hb = h[b, : len(c)].float()
            tot += (hb ** 2).sum().item()
            cnt += hb.numel()
    return math.sqrt(tot / cnt)


@torch.no_grad()
def encode_hiddens(encoder, items, device, pad_id=0,
                   autocast_dtype=torch.bfloat16):
    """Run frozen encoder on a batch of items -> list of (n_i, d) fp32."""
    maxn = max(it["n"] for it in items)
    ids = torch.full((len(items), maxn), pad_id, dtype=torch.long, device=device)
    for b, it in enumerate(items):
        ids[b, : it["n"]] = torch.tensor(it["tokens"], device=device)
    with torch.autocast("cuda", dtype=autocast_dtype):
        h = encoder(ids)
    return [h[b, : it["n"]].float() for b, it in enumerate(items)]
