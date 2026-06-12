"""Probe: (1) semsim calibration baselines, (2) out-of-domain reconstruction.

Answers: does small-K output really capture the gist (vs. random-pair
similarity)? And what happens on text far from wikitext-103?

Run: ../.venv/bin/python scripts/ood_probe.py   (writes results/task3_ood.md)
"""

import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, "src")
from data import ChunkStore, prepare_chunks  # noqa: E402
from model import ScratchLM  # noqa: E402
from tokenizer import load_tokenizer  # noqa: E402

OOD_TEXTS = [
    # casual dialogue (register OOD)
    "Hey, are you still coming to dinner tonight? I booked the Italian place "
    "near the station for 7:30, but I can push it back if you're stuck at work.",
    # news style (close-ish to wiki but present-tense reporting)
    "Stock markets fell sharply on Tuesday after the central bank signaled "
    "that interest rates would stay higher for longer than investors expected.",
    # recipe / instructions (procedural OOD)
    "Preheat the oven to 180 degrees. Mix the flour, sugar and butter until "
    "crumbly, then press the mixture into the tin and bake for 25 minutes.",
    # software docs (technical OOD)
    "To install the package, run pip install with the requirements file, then "
    "set the API key as an environment variable before starting the server.",
    # first-person narrative (register OOD)
    "I woke up before sunrise and walked down to the harbor. The boats were "
    "still dark, and the only sound was water slapping against the hulls.",
    # Japanese (extreme OOD: language unseen, byte-level BPE still encodes it)
    "今日の会議は午後3時からに変更になりました。資料は事前に共有フォルダに置いてください。",
]

K_SHOW = [1, 4, 16, 32, 64]


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    device = "cuda"
    torch.set_float32_matmul_precision("high")
    tok = load_tokenizer(cfg)
    n_max = cfg["data"]["n_max"]

    ck = torch.load("runs/enc_dct/ckpt.pt", map_location=device,
                    weights_only=False)
    model = ScratchLM(cfg, device, **ck["arch"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    # ---------- (1) semsim calibration on in-domain test ----------
    from sentence_transformers import SentenceTransformer

    semodel = SentenceTransformer(cfg["eval"]["semantic_model"], device=device)
    store = ChunkStore(prepare_chunks(cfg, tok)["test"])
    refs = [tok.decode(store.get(i).tolist()) for i in range(2000)]
    emb = semodel.encode(refs, batch_size=256, convert_to_tensor=True,
                         normalize_embeddings=True, show_progress_bar=False)
    rng = np.random.RandomState(0)
    perm = rng.permutation(2000)
    rand_pair = (emb * emb[perm]).sum(-1).mean().item()
    # adjacent chunks = same article, different content ("topic-level" anchor)
    adj = (emb[:-1] * emb[1:]).sum(-1).mean().item()
    print(f"semsim baselines: random pair {rand_pair:.3f} | "
          f"adjacent chunk (same article) {adj:.3f}")

    # ---------- (2) OOD reconstruction ----------
    lines = ["# OOD probe (enc_dct, greedy)", "",
             f"semsim baselines on in-domain test: random pair **{rand_pair:.3f}**, "
             f"adjacent chunks of the same article **{adj:.3f}**", ""]
    ood_sem = {k: [] for k in K_SHOW}
    ood_acc = {k: [] for k in K_SHOW}
    esc = lambda t: t.replace("|", "\\|").replace("\n", " ")  # noqa: E731
    for text in OOD_TEXTS:
        ids_list = tok.encode(text).ids[:n_max]
        n = len(ids_list)
        lines += [f"## n={n}: {esc(text[:60])}...", "", "| K | output |",
                  "|---|--------|", f"| orig | {esc(text)} |"]
        for k in K_SHOW:
            ke = min(k, n)
            ids = torch.zeros(1, n_max, dtype=torch.long, device=device)
            ids[0, :n] = torch.tensor(ids_list, device=device)
            lens = torch.tensor([n], device=device)
            idx = torch.arange(ke, device=device).unsqueeze(0)
            valid = torch.ones(1, ke, dtype=torch.bool, device=device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                gen = model.generate(ids, lens, idx, valid, mode="dct")
            out = tok.decode(gen[0, :n].tolist())
            acc = (gen[0, :n] == ids[0, :n]).float().mean().item()
            hv = semodel.encode([out, text], convert_to_tensor=True,
                                normalize_embeddings=True,
                                show_progress_bar=False)
            sem = (hv[0] * hv[1]).sum().item()
            ood_sem[k].append(sem)
            ood_acc[k].append(acc)
            lines.append(f"| {k} (acc {acc:.2f}, sem {sem:.2f}) | {esc(out)} |")
        lines.append("")
    lines += ["## OOD summary (mean over probes)", "",
              "| K | token_acc | semsim |", "|---|---|---|"]
    for k in K_SHOW:
        lines.append(f"| {k} | {np.mean(ood_acc[k]):.3f} | "
                     f"{np.mean(ood_sem[k]):.3f} |")
    open("results/task3_ood.md", "w").write("\n".join(lines))
    print("saved results/task3_ood.md")
    for k in K_SHOW:
        print(f"OOD K={k}: acc {np.mean(ood_acc[k]):.3f} "
              f"sem {np.mean(ood_sem[k]):.3f}")


if __name__ == "__main__":
    main()
