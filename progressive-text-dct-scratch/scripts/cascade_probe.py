"""Probe: is OOD failure missing information in Z, or autoregressive cascade?

Measures, at K=64 (information-lossless for n<=64):
  1. teacher-forced per-token accuracy (no cascade possible)
  2. free-run accuracy overall / before vs after the first error
on OOD probe texts and an in-domain test sample.

Run: ../.venv/bin/python scripts/cascade_probe.py
"""

import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, "src")
from data import ChunkStore, prepare_chunks  # noqa: E402
from model import ScratchLM  # noqa: E402
from tokenizer import load_tokenizer  # noqa: E402
from ood_probe import OOD_TEXTS  # noqa: E402


def probe(model, texts_ids, device, n_max, k=64):
    tf_acc, fr_acc, pre_err, post_err, first_err = [], [], [], [], []
    for ids_list in texts_ids:
        n = len(ids_list)
        ke = min(k, n)
        ids = torch.zeros(1, n_max, dtype=torch.long, device=device)
        ids[0, :n] = torch.tensor(ids_list, device=device)
        lens = torch.tensor([n], device=device)
        idx = torch.arange(ke, device=device).unsqueeze(0)
        valid = torch.ones(1, ke, dtype=torch.bool, device=device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, logits = model(ids[:, :n], lens, idx, valid, mode="dct",
                              return_logits=True)
            gen = model.generate(ids, lens, idx, valid, mode="dct")
        tf_pred = logits[0].float().argmax(-1)
        gt = ids[0, :n]
        tf_match = (tf_pred == gt)
        fr_match = (gen[0, :n] == gt)
        tf_acc.append(tf_match.float().mean().item())
        fr_acc.append(fr_match.float().mean().item())
        wrong = (~fr_match).nonzero()
        if len(wrong):
            fe = int(wrong[0])
            first_err.append(fe / n)
            if fe > 0:
                pre_err.append(1.0)  # by construction all correct before
            if fe < n - 1:
                post_err.append(fr_match[fe + 1 :].float().mean().item())
        else:
            first_err.append(1.0)
    return (np.mean(tf_acc), np.mean(fr_acc),
            np.mean(post_err) if post_err else float("nan"),
            np.mean(first_err))


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

    ood_ids = [tok.encode(t).ids[:n_max] for t in OOD_TEXTS]
    store = ChunkStore(prepare_chunks(cfg, tok)["test"])
    ind_ids = [store.get(i).tolist() for i in range(64) if len(store.get(i)) <= 64]

    for name, tids in (("in-domain (n<=64)", ind_ids), ("OOD", ood_ids)):
        tf, fr, post, fe = probe(model, tids, device, n_max)
        print(f"{name:18s} K=64: teacher-forced acc {tf:.3f} | free-run acc "
              f"{fr:.3f} | acc after 1st error {post:.3f} | "
              f"1st error at {fe*100:.0f}% of length")


if __name__ == "__main__":
    main()
