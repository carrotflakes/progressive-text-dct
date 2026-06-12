"""From-scratch prefix-LM Transformer decoder + end-to-end DCT compressor.

Sequence layout (P = 1 + k_pad):
    pos 0        : length embedding
    pos 1..k_pad : projected DCT coefficients + coefficient-index embeddings
                   (slots >= k_b are padding, masked out)
    pos P        : BOS (learned vector)
    pos P+1..P+n : teacher-forced text tokens t_1..t_n (decoder's own embedding)

Attention (prefix-LM): prefix positions attend bidirectionally within the
prefix; BOS/text positions attend to the prefix plus causally among
themselves. Logits at positions P..P+n-1 predict t_1..t_n.

Everything is built with batched tensor ops (no per-sample Python loops) —
the task-1 run was launch-overhead-bound, so this is the main speed lesson.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dct import BatchDCT


# ---------------------------------------------------------------- RoPE

def rope_cache(max_len, head_dim, base, device):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device) / head_dim))
    t = torch.arange(max_len, device=device)
    freqs = torch.outer(t, inv)  # (max_len, head_dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin, pos0=0):
    """x: (B, H, T, hd). Rotate pairs (even, odd) by position."""
    T = x.shape[2]
    c = cos[pos0 : pos0 + T].unsqueeze(0).unsqueeze(0)  # (1,1,T,hd/2)
    s = sin[pos0 : pos0 + T].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out


# ---------------------------------------------------------------- blocks

class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, n_layers):
        super().__init__()
        self.n_heads = n_heads
        self.hd = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.attn_out = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
        # GPT-2 style scaled init for residual-out projections
        for p in (self.attn_out.weight, self.ff2.weight):
            nn.init.normal_(p, std=0.02 / math.sqrt(2 * n_layers))

    def forward(self, x, mask, cos, sin, pos0=0, past=None):
        """past: optional (k, v) of shape (B, H, T_past, hd); returns new past."""
        B, T, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.hd).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.hd).transpose(1, 2)
        q = apply_rope(q, cos, sin, pos0)
        k = apply_rope(k, cos, sin, pos0)
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        new_past = (k, v)
        # mask: (B, 1, T_q, T_kv) bool, True = attend
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).reshape(B, T, -1)
        x = x + self.drop(self.attn_out(y))
        x = x + self.drop(self.ff2(F.gelu(self.ff1(self.ln2(x)))))
        return x, new_past


class CrossBlock(nn.Module):
    """Latent update block: cross-attention to encoder output + latent
    self-attention + FFN (Perceiver-style, pre-LN)."""

    def __init__(self, d, n_heads, d_ff):
        super().__init__()
        self.n_heads = n_heads
        self.hd = d // n_heads
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)
        self.q_proj = nn.Linear(d, d)
        self.kv_proj = nn.Linear(d, 2 * d)
        self.cross_out = nn.Linear(d, d)
        self.ln_s = nn.LayerNorm(d)
        self.self_qkv = nn.Linear(d, 3 * d)
        self.self_out = nn.Linear(d, d)
        self.ln_f = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, d_ff)
        self.ff2 = nn.Linear(d_ff, d)

    def _heads(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.hd).transpose(1, 2)

    def forward(self, lat, enc, key_mask):
        """lat (B, M, d); enc (B, n, d); key_mask (B, 1, 1, n) bool."""
        q = self._heads(self.q_proj(self.ln_q(lat)))
        k, v = self.kv_proj(self.ln_kv(enc)).chunk(2, dim=-1)
        y = F.scaled_dot_product_attention(q, self._heads(k), self._heads(v),
                                           attn_mask=key_mask)
        lat = lat + self.cross_out(y.transpose(1, 2).reshape_as(lat))
        q, k, v = self.self_qkv(self.ln_s(lat)).chunk(3, dim=-1)
        y = F.scaled_dot_product_attention(
            self._heads(q), self._heads(k), self._heads(v))
        lat = lat + self.self_out(y.transpose(1, 2).reshape_as(lat))
        return lat + self.ff2(F.gelu(self.ff1(self.ln_f(lat))))


class ScratchLM(nn.Module):
    """Decoder transformer + all conditioning embeddings + encoder table E.

    Optional (task3): a bidirectional Transformer contextualizer over E before
    the bottleneck (`encoder="transformer"`), and a learned-latent bottleneck
    (`bottleneck="latent"`, mode="latent") instead of DCT.
    """

    def __init__(self, cfg, device, encoder="none", bottleneck="dct"):
        super().__init__()
        V = cfg["tokenizer"]["vocab_size"]
        d_emb = cfg["compress"]["d_emb"]
        m = cfg["model"]
        d = m["d_model"]
        self.n_max = cfg["data"]["n_max"]
        self.k_max = cfg["compress"]["k_max"]
        self.d_emb = d_emb
        self.arch = {"encoder": encoder, "bottleneck": bottleneck}

        # --- encoder side (the compressed representation lives here) ---
        self.enc_emb = nn.Embedding(V, d_emb)            # E, trained end-to-end
        ecfg = cfg.get("encoder", {})
        if encoder == "transformer":
            self.enc_blocks = nn.ModuleList(
                Block(d_emb, m["n_heads"], m["d_ff"], m["dropout"],
                      ecfg.get("n_layers", 4))
                for _ in range(ecfg.get("n_layers", 4)))
            self.enc_ln = nn.LayerNorm(d_emb)
        else:
            self.enc_blocks = None
        if bottleneck == "latent":
            n_lat = ecfg.get("n_latents", 64)
            self.latent_q = nn.Parameter(torch.randn(n_lat, d_emb) * 0.02)
            self.cross_blocks = nn.ModuleList(
                CrossBlock(d_emb, m["n_heads"], m["d_ff"])
                for _ in range(ecfg.get("cross_blocks", 2)))

        # --- decoder side ---
        self.proj = nn.Linear(d_emb, d)
        self.idx_emb = nn.Embedding(self.n_max, d)       # coefficient index
        self.len_emb = nn.Embedding(self.n_max + 1, d)   # original length n
        self.bos = nn.Parameter(torch.zeros(d))
        self.tok_emb = nn.Embedding(V, d)                # independent of E
        self.blocks = nn.ModuleList(
            Block(d, m["n_heads"], m["d_ff"], m["dropout"], m["n_layers"])
            for _ in range(m["n_layers"]))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)

        for mod in self.modules():
            if isinstance(mod, nn.Embedding):
                nn.init.normal_(mod.weight, std=0.02)
        nn.init.normal_(self.bos, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)

        max_pos = 2 + self.n_max + self.n_max  # len + z slots + bos + text
        cos, sin = rope_cache(max_pos, d // m["n_heads"], m["rope_base"], device)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.dct = BatchDCT(self.n_max, device)

    # ---------------- compression (differentiable, no detach!) ----------------

    def encode(self, ids, lens):
        """E lookup, optionally contextualized by the bidirectional encoder.
        Returns (B, T, d_emb) with padding rows zeroed."""
        valid = (torch.arange(ids.shape[1], device=ids.device)[None, :]
                 < lens[:, None])
        h = self.enc_emb(ids) * valid.unsqueeze(-1)
        if self.enc_blocks is not None:
            key_mask = valid[:, None, None, :]      # bidirectional, valid keys
            for blk in self.enc_blocks:
                h, _ = blk(h, key_mask, self.rope_cos, self.rope_sin)
            h = self.enc_ln(h) * valid.unsqueeze(-1)
        return h

    def compress(self, ids, lens, idx, idx_valid, mode="dct"):
        """ids (B, <=n_max) padded token ids; lens (B,); idx (B, k_pad) kept
        coefficient/latent indices; idx_valid (B, k_pad) bool.
        Returns Z (B, k_pad, d_emb)."""
        h = self.encode(ids, lens)
        if mode == "latent":
            valid = (torch.arange(ids.shape[1], device=ids.device)[None, :]
                     < lens[:, None])
            lat = self.latent_q.unsqueeze(0).expand(ids.shape[0], -1, -1)
            for cb in self.cross_blocks:
                lat = cb(lat, h, valid[:, None, None, :])
            zfull = lat                              # (B, M, d_emb)
        elif mode == "dct":
            if h.shape[1] < self.n_max:             # DCT matrices are (n_max, n_max)
                h = F.pad(h, (0, 0, 0, self.n_max - h.shape[1]))
            zfull = self.dct.forward(h.float(), lens).to(h.dtype)
        else:                                       # "trunc" (B4)
            zfull = h
        z = torch.gather(zfull, 1,
                         idx.unsqueeze(-1).expand(-1, -1, self.d_emb))
        return z * idx_valid.unsqueeze(-1)

    # ---------------- sequence assembly ----------------

    def build_prefix(self, z, lens, idx, idx_valid):
        """-> prefix hidden states (B, P, d), P = 1 + k_pad (+BOS appended later)."""
        zp = self.proj(z) + self.idx_emb(idx) * idx_valid.unsqueeze(-1)
        le = self.len_emb(lens).unsqueeze(1)        # (B, 1, d)
        return torch.cat([le, zp], dim=1)

    @staticmethod
    def prefix_lm_mask(k_pad, n_pad, idx_valid, text_valid):
        """(B, 1, L, L) bool mask. L = 1 + k_pad + 1 + n_pad."""
        B = idx_valid.shape[0]
        dev = idx_valid.device
        P = 1 + k_pad
        L = P + 1 + n_pad
        # key validity
        kv = torch.ones(B, L, dtype=torch.bool, device=dev)
        kv[:, 1 : 1 + k_pad] = idx_valid
        kv[:, P + 1 :] = text_valid
        is_prefix = torch.zeros(L, dtype=torch.bool, device=dev)
        is_prefix[:P] = True
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev))
        # prefix query -> prefix keys (bidirectional); text query -> prefix + causal
        struct = torch.where(is_prefix[:, None],
                             is_prefix[None, :].expand(L, L),
                             is_prefix[None, :] | causal)
        mask = struct[None, :, :] & kv[:, None, :]
        mask = mask | torch.eye(L, dtype=torch.bool, device=dev)[None]
        return mask.unsqueeze(1)

    # ---------------- training forward ----------------

    def forward(self, ids, lens, idx, idx_valid, mode="dct"):
        """Teacher-forced CE loss. ids (B, n_max) padded with -1 -> masked."""
        B = ids.shape[0]
        dev = ids.device
        n_pad = int(lens.max().item())
        k_pad = idx.shape[1]
        text_valid = (torch.arange(n_pad, device=dev)[None, :] < lens[:, None])
        ids_in = ids[:, :n_pad].clamp(min=0)

        z = self.compress(ids_in * text_valid, lens, idx, idx_valid, mode)
        prefix = self.build_prefix(z, lens, idx, idx_valid)
        bos = self.bos[None, None, :].expand(B, 1, -1)
        text = self.tok_emb(ids_in) * text_valid.unsqueeze(-1)
        x = torch.cat([prefix, bos, text], dim=1)
        mask = self.prefix_lm_mask(k_pad, n_pad, idx_valid, text_valid)

        for blk in self.blocks:
            x, _ = blk(x, mask, self.rope_cos, self.rope_sin)
        P = 1 + k_pad
        # logits at positions P..P+n-1 predict t_1..t_n
        hs = self.ln_f(x[:, P : P + n_pad])
        logits = self.head(hs)
        labels = torch.where(text_valid, ids_in, torch.full_like(ids_in, -100))
        loss = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(),
                               ignore_index=-100)
        return loss

    # ---------------- generation (greedy, KV cache) ----------------

    @torch.no_grad()
    def generate(self, ids, lens, idx, idx_valid, mode="dct"):
        """Greedy-decode exactly lens[b] tokens. Returns (B, n_pad) token ids."""
        B = ids.shape[0]
        dev = ids.device
        n_pad = int(lens.max().item())
        k_pad = idx.shape[1]
        text_valid = (torch.arange(n_pad, device=dev)[None, :] < lens[:, None])
        ids_in = ids[:, :n_pad].clamp(min=0)

        z = self.compress(ids_in * text_valid, lens, idx, idx_valid, mode)
        prefix = self.build_prefix(z, lens, idx, idx_valid)
        bos = self.bos[None, None, :].expand(B, 1, -1)
        x = torch.cat([prefix, bos], dim=1)           # (B, P+1, d)
        P = 1 + k_pad

        # prefill: prefix-LM mask restricted to the first P+1 positions
        kv = torch.ones(B, P + 1, dtype=torch.bool, device=dev)
        kv[:, 1 : 1 + k_pad] = idx_valid
        is_prefix = torch.zeros(P + 1, dtype=torch.bool, device=dev)
        is_prefix[:P] = True
        struct = torch.where(
            is_prefix[:, None], is_prefix[None, :].expand(P + 1, P + 1),
            is_prefix[None, :] | torch.tril(
                torch.ones(P + 1, P + 1, dtype=torch.bool, device=dev)))
        mask = (struct[None] & kv[:, None, :]).unsqueeze(1)
        pasts = []
        for blk in self.blocks:
            x, past = blk(x, mask, self.rope_cos, self.rope_sin, pos0=0)
            pasts.append(past)
        tok = self.head(self.ln_f(x[:, -1])).argmax(-1)   # t_1

        out = torch.zeros(B, n_pad, dtype=torch.long, device=dev)
        out[:, 0] = tok
        kv_step = kv  # growing key-validity vector
        for t in range(1, n_pad):
            xt = self.tok_emb(tok).unsqueeze(1)
            kv_step = torch.cat(
                [kv_step, torch.ones(B, 1, dtype=torch.bool, device=dev)], dim=1)
            mask_t = kv_step[:, None, None, :]            # (B,1,1,T_kv)
            new_pasts = []
            for blk, past in zip(self.blocks, pasts):
                xt, npast = blk(xt, mask_t, self.rope_cos, self.rope_sin,
                                pos0=P + t, past=past)
                new_pasts.append(npast)
            pasts = new_pasts
            tok = self.head(self.ln_f(xt[:, -1])).argmax(-1)
            out[:, t] = tok
        return out * text_valid
