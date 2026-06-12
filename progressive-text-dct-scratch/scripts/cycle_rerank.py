"""Inference-time test of the cycle-consistency idea (no training):
sample N candidates, re-encode each, keep the one whose truncated DCT
coefficients are closest to the original's (cosine in Z-space).

If this beats greedy on fluency/semantics at small K, the decoder already
knows fluent candidates and a cycle objective is well-motivated.

Metrics per K: token acc, semsim (MiniLM), fluency PPL (GPT-2, eval-only).
Run: ../.venv/bin/python scripts/cycle_rerank.py
"""

import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, "src")
from data import ChunkStore, eval_batch, prepare_chunks  # noqa: E402
from model import ScratchLM  # noqa: E402
from tokenizer import load_tokenizer  # noqa: E402

KS = [1, 4, 16]
N_PROBE = 256
N_SAMPLES = 8       # sampled candidates (greedy is added as one more)
TEMP = 0.8
BS = 256


@torch.no_grad()
def gpt2_ppl(texts, device):
    """Mean per-token NLL under pretrained GPT-2 (evaluation-only)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained("gpt2",
                                              dtype=torch.float32).to(device)
    lm.eval()
    nll, cnt = 0.0, 0
    for s in range(0, len(texts), 32):
        batch = [t if t.strip() else "." for t in texts[s : s + 32]]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=256)
        ids = enc["input_ids"].to(device)
        am = enc["attention_mask"].to(device)
        out = lm(input_ids=ids, attention_mask=am)
        lp = torch.log_softmax(out.logits[:, :-1].float(), -1)
        tgt = ids[:, 1:]
        ll = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        m = am[:, 1:].bool()
        nll += float(-(ll * m).sum())
        cnt += int(m.sum())
    del lm
    torch.cuda.empty_cache()
    return float(np.exp(nll / cnt))


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    device = "cuda"
    torch.set_float32_matmul_precision("high")
    tok = load_tokenizer(cfg)
    n_max = cfg["data"]["n_max"]
    store = ChunkStore(prepare_chunks(cfg, tok)["test"])

    ck = torch.load("runs/enc_dct/ckpt.pt", map_location=device,
                    weights_only=False)
    model = ScratchLM(cfg, device, **ck["arch"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    from sentence_transformers import SentenceTransformer

    semodel = SentenceTransformer(cfg["eval"]["semantic_model"], device=device)
    refs = [tok.decode(store.get(i).tolist()) for i in range(N_PROBE)]
    ref_emb = semodel.encode(refs, batch_size=256, convert_to_tensor=True,
                             normalize_embeddings=True, show_progress_bar=False)

    rows = []
    for k in KS:
        ids, lens, idx, valid = eval_batch(store, list(range(N_PROBE)), k,
                                           "main", cfg["seed"], n_max, device)
        n_pad = int(lens.max())
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            z0 = model.compress(ids[:, :n_pad], lens, idx, valid, mode="dct")
        z0f = torch.nn.functional.normalize(z0.float().flatten(1), dim=-1)

        cands, scores = [], []
        for s in range(N_SAMPLES + 1):
            t = 0.0 if s == 0 else TEMP
            torch.manual_seed(1000 + s)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                gen = model.generate(ids, lens, idx, valid, mode="dct",
                                     temperature=t)
                zc = model.compress(gen[:, :n_pad], lens, idx, valid,
                                    mode="dct")
            zcf = torch.nn.functional.normalize(zc.float().flatten(1), dim=-1)
            cands.append(gen)
            scores.append((z0f * zcf).sum(-1))   # cycle score per sample
        scores = torch.stack(scores)             # (S+1, N)
        best = scores.argmax(0)                  # (N,)

        def metrics(pick_fn, name):
            outs, accs = [], []
            for i in range(N_PROBE):
                g = pick_fn(i)
                n = int(lens[i])
                accs.append(float((g[:n] == ids[i, :n]).float().mean()))
                outs.append(tok.decode(g[:n].tolist()))
            emb = semodel.encode(outs, batch_size=256, convert_to_tensor=True,
                                 normalize_embeddings=True,
                                 show_progress_bar=False)
            sem = float((emb * ref_emb).sum(-1).mean())
            ppl = gpt2_ppl(outs, device)
            rows.append([k, name, np.mean(accs), sem, ppl])
            print(f"K={k} {name:12s}: acc {np.mean(accs):.3f} sem {sem:.3f} "
                  f"fluencyPPL {ppl:7.1f}", flush=True)
            return emb

        metrics(lambda i: cands[0][i], "greedy")
        metrics(lambda i: cands[best[i]][i], "cycle-rerank")
        # oracle: pick candidate with highest true semsim (headroom)
        all_emb = []
        for s in range(N_SAMPLES + 1):
            outs = [tok.decode(cands[s][i, : int(lens[i])].tolist())
                    for i in range(N_PROBE)]
            all_emb.append(semodel.encode(
                outs, batch_size=256, convert_to_tensor=True,
                normalize_embeddings=True, show_progress_bar=False))
        sims = torch.stack([(e * ref_emb).sum(-1) for e in all_emb])  # (S+1,N)
        ob = sims.argmax(0)
        metrics(lambda i: cands[ob[i]][i], "oracle-sem")

    print("\nref fluencyPPL:", f"{gpt2_ppl(refs, device):.1f}")
    with open("results/task3_cycle_rerank.csv", "w") as f:
        f.write("K,method,token_acc,semsim,fluency_ppl\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print("saved results/task3_cycle_rerank.csv")


if __name__ == "__main__":
    main()
