"""Re-examine semantic preservation of the bottleneck REPRESSENTATIONS,
bypassing the autoregressive decoder (which we showed cascades).

Probe 1: teacher-forced token accuracy per K (readable info, no cascade)
Probe 2: representational similarity analysis (RSA): Spearman correlation
         between cosine similarities in flattened Z-space (K x 512) and in
         MiniLM semantic space, over test pairs. No decoder involved.

Run: ../.venv/bin/python scripts/latent_semantics.py
"""

import sys

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr

sys.path.insert(0, "src")
from data import ChunkStore, eval_batch, prepare_chunks  # noqa: E402
from model import ScratchLM  # noqa: E402
from tokenizer import load_tokenizer  # noqa: E402

KS = [1, 2, 4, 8, 16, 32, 64]
N_PROBE = 512
BS = 128


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    device = "cuda"
    torch.set_float32_matmul_precision("high")
    tok = load_tokenizer(cfg)
    n_max = cfg["data"]["n_max"]
    store = ChunkStore(prepare_chunks(cfg, tok)["test"])

    from sentence_transformers import SentenceTransformer

    semodel = SentenceTransformer(cfg["eval"]["semantic_model"], device=device)
    refs = [tok.decode(store.get(i).tolist()) for i in range(N_PROBE)]
    sem_emb = semodel.encode(refs, batch_size=256, convert_to_tensor=True,
                             normalize_embeddings=True, show_progress_bar=False)
    sem_sims = (sem_emb @ sem_emb.T).cpu().numpy()
    iu = np.triu_indices(N_PROBE, k=1)
    sem_flat = sem_sims[iu]

    results = {}
    for variant in ("enc_dct", "enc_latent", "emb_dct"):
        ck = torch.load(f"runs/{variant}/ckpt.pt", map_location=device,
                        weights_only=False)
        model = ScratchLM(cfg, device, **ck["arch"]).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        mode = ck["mode"]
        for k in KS:
            tf_match = tf_total = 0
            zs = []
            for s in range(0, N_PROBE, BS):
                idxs = list(range(s, min(s + BS, N_PROBE)))
                ids, lens, idx, valid = eval_batch(store, idxs, k, "main",
                                                   cfg["seed"], n_max, device)
                n_pad = int(lens.max())
                with torch.no_grad(), torch.autocast("cuda",
                                                     dtype=torch.bfloat16):
                    _, logits = model(ids[:, :n_pad], lens, idx, valid,
                                      mode=mode, return_logits=True)
                    z = model.compress(ids[:, :n_pad], lens, idx, valid,
                                       mode=mode)
                pred = logits.float().argmax(-1)
                tv = (torch.arange(n_pad, device=device)[None, :]
                      < lens[:, None])
                tf_match += int(((pred == ids[:, :n_pad]) & tv).sum())
                tf_total += int(tv.sum())
                # pad Z to K columns (samples with n<k have fewer) and flatten
                if z.shape[1] < k:
                    z = torch.nn.functional.pad(z, (0, 0, 0, k - z.shape[1]))
                zs.append(z.float().flatten(1).cpu())
            zf = torch.nn.functional.normalize(torch.cat(zs), dim=-1)
            z_sims = (zf @ zf.T).numpy()[iu]
            rsa = spearmanr(z_sims, sem_flat).statistic
            results[(variant, k)] = (tf_match / tf_total, rsa)
            print(f"{variant:11s} K={k:2d}: TF acc {tf_match/tf_total:.3f} "
                  f"RSA {rsa:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()

    lines = ["# Semantic preservation of representations (decoder-free probes)",
             "", f"N={N_PROBE} test chunks. TF = teacher-forced token acc; "
             "RSA = Spearman(Z-space sims, MiniLM sims).", "",
             "| K | " + " | ".join(f"{v} TF / RSA"
                                   for v in ("enc_dct", "enc_latent",
                                             "emb_dct")) + " |",
             "|---|---|---|---|"]
    for k in KS:
        row = f"| {k} |"
        for v in ("enc_dct", "enc_latent", "emb_dct"):
            tf, rsa = results[(v, k)]
            row += f" {tf:.3f} / {rsa:.3f} |"
        lines.append(row)
    open("results/task3_latent_semantics.md", "w").write("\n".join(lines))
    print("saved results/task3_latent_semantics.md")


if __name__ == "__main__":
    main()
