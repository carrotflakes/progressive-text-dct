"""Rerank v2 on the K=0-mixed model (enc_dct_k0): combine cycle score with the
model's OWN unconditional (K=0) likelihood as a self-contained fluency term.

Candidate score = z(cycle cosine) + lam * z(-K0 NLL), z-scored per sample
over the candidate pool. Compares greedy / cycle-only / fluency-only /
combined / oracle-sem, with GPT-2 PPL (eval-only) as the external fluency
check.

Run: ../.venv/bin/python scripts/cycle_rerank2.py
"""

import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from data import ChunkStore, eval_batch, prepare_chunks  # noqa: E402
from model import ScratchLM  # noqa: E402
from tokenizer import load_tokenizer  # noqa: E402
from cycle_rerank import gpt2_ppl  # noqa: E402

KS = [1, 4, 16]
N_PROBE = 256
N_SAMPLES = 8
TEMP = 0.8
LAM = 1.0


@torch.no_grad()
def k0_nll(model, gen, lens, n_pad, device):
    """Per-sample mean NLL of token rows under the model's K=0 (LM) mode."""
    B = gen.shape[0]
    idx0 = torch.zeros(B, 0, dtype=torch.long, device=device)
    val0 = torch.zeros(B, 0, dtype=torch.bool, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, logits = model(gen[:, :n_pad], lens, idx0, val0, mode="dct",
                          return_logits=True)
    lp = torch.log_softmax(logits.float(), -1)
    tok_lp = lp.gather(-1, gen[:, :n_pad].clamp(min=0).unsqueeze(-1)).squeeze(-1)
    mask = (torch.arange(n_pad, device=device)[None, :] < lens[:, None]).float()
    return -(tok_lp * mask).sum(1) / lens.float()


def zscore(x):
    return (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + 1e-6)


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    device = "cuda"
    torch.set_float32_matmul_precision("high")
    tok = load_tokenizer(cfg)
    n_max = cfg["data"]["n_max"]
    store = ChunkStore(prepare_chunks(cfg, tok)["test"])

    ck = torch.load("runs/enc_dct_k0/ckpt.pt", map_location=device,
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

        cands, cyc, flu = [], [], []
        for s in range(N_SAMPLES + 1):
            t = 0.0 if s == 0 else TEMP
            torch.manual_seed(2000 + s)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                gen = model.generate(ids, lens, idx, valid, mode="dct",
                                     temperature=t)
                zc = model.compress(gen[:, :n_pad], lens, idx, valid,
                                    mode="dct")
            zcf = torch.nn.functional.normalize(zc.float().flatten(1), dim=-1)
            cands.append(gen)
            cyc.append((z0f * zcf).sum(-1))
            flu.append(-k0_nll(model, gen, lens, n_pad, device))
        cyc = torch.stack(cyc)          # (S+1, N)
        flu = torch.stack(flu)
        combined = zscore(cyc) + LAM * zscore(flu)

        def metrics(pick, name):
            outs, accs = [], []
            for i in range(N_PROBE):
                g = cands[pick[i]][i]
                n = int(lens[i])
                accs.append(float((g[:n] == ids[i, :n]).float().mean()))
                outs.append(tok.decode(g[:n].tolist()))
            emb = semodel.encode(outs, batch_size=256, convert_to_tensor=True,
                                 normalize_embeddings=True,
                                 show_progress_bar=False)
            sem = float((emb * ref_emb).sum(-1).mean())
            ppl = gpt2_ppl(outs, device)
            rows.append([k, name, float(np.mean(accs)), sem, ppl])
            print(f"K={k} {name:13s}: acc {np.mean(accs):.3f} sem {sem:.3f} "
                  f"fluencyPPL {ppl:7.1f}", flush=True)

        zero = torch.zeros(N_PROBE, dtype=torch.long)
        metrics(zero, "greedy")
        metrics(cyc.argmax(0), "cycle-only")
        metrics(flu.argmax(0), "fluency-only")
        metrics(combined.argmax(0), "cycle+fluency")
        all_emb = []
        for s in range(N_SAMPLES + 1):
            outs = [tok.decode(cands[s][i, : int(lens[i])].tolist())
                    for i in range(N_PROBE)]
            all_emb.append(semodel.encode(
                outs, batch_size=256, convert_to_tensor=True,
                normalize_embeddings=True, show_progress_bar=False))
        sims = torch.stack([(e * ref_emb).sum(-1) for e in all_emb])
        metrics(sims.argmax(0).cpu(), "oracle-sem")

    with open("results/task4_cycle_rerank2.csv", "w") as f:
        f.write("K,method,token_acc,semsim,fluency_ppl\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print("saved results/task4_cycle_rerank2.csv")


if __name__ == "__main__":
    main()
