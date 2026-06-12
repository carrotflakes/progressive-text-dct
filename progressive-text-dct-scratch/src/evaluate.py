"""Evaluate all trained variants + B1 (training-free NN decode) on test-2000.

Per K in eval.k_values: token accuracy, exact match, BLEU, chrF (sacrebleu),
semantic cosine (all-MiniLM-L6-v2, evaluation-only). Also produces the H3
spectrum analysis (DCT energy distribution of learned vs frozen-random E).

Outputs: results/metrics.csv, curves.png, spectrum.png, samples.md
Usage: python src/evaluate.py --config config.yaml [--variants main,b2,b3,b4,b1]
"""

import argparse
import csv
import os

import numpy as np
import torch
import yaml

from data import ChunkStore, eval_batch, prepare_chunks
from tokenizer import load_tokenizer
from model import ScratchLM

LABEL = {"main": "main (DCT, learned E)", "b1": "B1 untrained NN decode",
         "b2": "B2 frozen-random E", "b3": "B3 random-K coeffs",
         "b4": "B4 prefix truncation",
         "enc_dct": "A: encoder + DCT", "enc_latent": "B: encoder + latents",
         "emb_dct": "C: embedding-only DCT"}


def batches(n, bs):
    for s in range(0, n, bs):
        yield list(range(s, min(s + bs, n)))


def token_metrics(gen, ids, lens):
    """gen, ids: (B, *) tensors; lens: (B,)."""
    match = total = exact = 0
    for j in range(len(lens)):
        n = int(lens[j])
        m = int((gen[j, :n] == ids[j, :n]).sum())
        match += m
        total += n
        exact += int(m == n)
    return match, total, exact


def decode_rows(tok, rows, lens):
    return [tok.decode(rows[j, : int(lens[j])].tolist()) for j in range(len(lens))]


@torch.no_grad()
def eval_b1(model, ids, lens, idx, valid):
    """Zero-fill kept coefficients -> inverse DCT -> cosine NN in learned E."""
    h = model.enc_emb(ids)
    nmask = (torch.arange(ids.shape[1], device=ids.device)[None, :, None]
             < lens[:, None, None])
    h = (h * nmask).float()
    z = model.dct.forward(h, lens)
    keep = torch.zeros_like(z)
    keep.scatter_(1, idx.unsqueeze(-1).expand(-1, -1, z.shape[-1]),
                  torch.gather(z, 1, idx.unsqueeze(-1).expand(-1, -1, z.shape[-1]))
                  * valid.unsqueeze(-1))
    h_rec = model.dct.inverse(keep, lens)
    e_n = torch.nn.functional.normalize(model.enc_emb.weight.float(), dim=-1)
    h_n = torch.nn.functional.normalize(h_rec, dim=-1)
    return (h_n @ e_n.T).argmax(-1) * (nmask.squeeze(-1))


def spectrum_analysis(models, store, cfg, out_png, device, n_samples=2000):
    """Mean DCT power per coefficient index of the pre-bottleneck
    representation (raw E or encoder output)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_max = cfg["data"]["n_max"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for name, model in models.items():
        power = torch.zeros(n_max, device=device)
        count = torch.zeros(n_max, device=device)
        with torch.no_grad():
            for idxs in batches(min(n_samples, len(store)), 256):
                ids, lens, idx, valid = eval_batch(
                    store, idxs, n_max, "main", cfg["seed"], n_max, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    h = model.encode(ids, lens)
                h = h.float()
                z = model.dct.forward(h, lens)         # (B, n_max, d)
                p = (z ** 2).mean(-1)                  # (B, n_max)
                fmask = (torch.arange(n_max, device=device)[None, :]
                         < lens[:, None]).float()
                power += (p * fmask).sum(0)
                count += fmask.sum(0)
        spec = (power / count.clamp(min=1)).cpu().numpy()
        axes[0].plot(spec, label=name)
        cum = np.cumsum(spec) / spec.sum()
        axes[1].plot(cum, label=name)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("DCT coefficient index (frequency)")
    axes[0].set_ylabel("mean power")
    axes[0].set_title("DCT energy spectrum of embedded text")
    axes[1].set_xlabel("DCT coefficient index")
    axes[1].set_ylabel("cumulative energy fraction")
    axes[1].axvline(64, color="gray", ls="--", lw=0.8)
    axes[1].set_title("Cumulative energy (K_max=64 dashed)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"saved {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--variants", default="main,b2,b3,b4,b1")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="",
                    help="prefix for output filenames (e.g. task3_)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    os.makedirs(args.out, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    device = "cuda"
    tok = load_tokenizer(cfg)
    store = ChunkStore(prepare_chunks(cfg, tok)["test"])
    n_test = min(args.limit or len(store), len(store))
    n_max = cfg["data"]["n_max"]
    gen_bs = cfg["eval"]["gen_batch_size"]
    k_values = cfg["eval"]["k_values"]

    from sentence_transformers import SentenceTransformer
    semodel = SentenceTransformer(cfg["eval"]["semantic_model"], device=device)
    import sacrebleu

    ref_texts = None
    rows = []
    gen_store = {}
    spec_models = {}

    def load_model(variant):
        ck = torch.load(os.path.join(args.runs, variant, "ckpt.pt"),
                        map_location=device, weights_only=False)
        arch = ck.get("arch", {"encoder": "none", "bottleneck": "dct"})
        m = ScratchLM(cfg, device, **arch).to(device)
        m.load_state_dict(ck["model"])
        m.eval()
        return m, ck["mode"]

    spectrum_variants = ("main", "b2", "enc_dct", "emb_dct")
    for variant in args.variants.split(","):
        src_variant = "main" if variant == "b1" else variant
        ck_path = os.path.join(args.runs, src_variant, "ckpt.pt")
        if not os.path.exists(ck_path):
            print(f"skip {variant}: no {ck_path}")
            continue
        model, mode = load_model(src_variant)
        if variant in spectrum_variants:
            spec_models[LABEL[variant]] = model
        for k in k_values:
            match = total = exact = 0
            hyps, refs = [], []
            for idxs in batches(n_test, gen_bs):
                ids, lens, idx, valid = eval_batch(
                    store, idxs, k, "b3" if variant == "b3" else "main",
                    cfg["seed"], n_max, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if variant == "b1":
                        gen = eval_b1(model, ids, lens, idx, valid)
                    else:
                        gen = model.generate(ids, lens, idx, valid, mode=mode)
                m_, t_, e_ = token_metrics(gen, ids, lens)
                match += m_; total += t_; exact += e_
                hyps += decode_rows(tok, gen.cpu(), lens)
                refs += decode_rows(tok, ids.cpu(), lens)
            if ref_texts is None:
                ref_texts = refs
            acc = match / total
            em = exact / n_test
            bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
            chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
            hv = semodel.encode(hyps, batch_size=256, convert_to_tensor=True,
                                normalize_embeddings=True, show_progress_bar=False)
            rv = semodel.encode(refs, batch_size=256, convert_to_tensor=True,
                                normalize_embeddings=True, show_progress_bar=False)
            sem = (hv * rv).sum(-1).mean().item()
            rows.append([variant, k, acc, em, bleu, chrf, sem])
            gen_store[(variant, k)] = hyps
            print(f"{variant} K={k}: acc={acc:.4f} em={em:.4f} bleu={bleu:.2f} "
                  f"chrf={chrf:.2f} sem={sem:.4f}", flush=True)
        if variant not in spectrum_variants:
            del model
            torch.cuda.empty_cache()

    tag = args.tag
    with open(os.path.join(args.out, f"{tag}metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "K", "token_acc", "exact_match", "bleu", "chrf",
                    "semsim"])
        w.writerows(rows)

    qual_variant = args.variants.split(",")[0]
    plot_curves(rows, os.path.join(args.out, f"{tag}curves.png"))
    write_samples(gen_store, store, tok, cfg,
                  os.path.join(args.out, f"{tag}samples.md"), n_test,
                  qual_variant)
    if spec_models:
        spectrum_analysis(spec_models, store, cfg,
                          os.path.join(args.out, f"{tag}spectrum.png"), device)
    print("done")


def plot_curves(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = {"token_acc": 2, "chrf": 5, "semsim": 6}
    order = ["main", "b1", "b2", "b3", "b4", "enc_dct", "enc_latent", "emb_dct"]
    variants = sorted({r[0] for r in rows},
                      key=lambda v: order.index(v) if v in order else 9)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (mkey, title) in zip(axes, [("token_acc", "Token accuracy"),
                                        ("chrf", "chrF"),
                                        ("semsim", "Semantic cosine sim")]):
        for v in variants:
            pts = sorted([(r[1], r[cols[mkey]]) for r in rows if r[0] == v])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                    label=LABEL.get(v, v))
        ax.set_xscale("log", base=2)
        ax.set_xlabel("K (kept coefficients)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Reconstruction quality vs K (from-scratch)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


def write_samples(gen_store, store, tok, cfg, path, n_test, qual_variant="main"):
    qks = cfg["eval"]["qualitative_ks"]
    qn = cfg["eval"]["qualitative_n"]
    others = sorted({v for v, _ in gen_store} - {qual_variant})
    picked = [i for i in range(n_test) if 40 <= len(store.get(i)) <= 90][:qn]
    esc = lambda t: t.replace("|", "\\|").replace("\n", " ⏎ ")  # noqa: E731
    lines = [f"# Qualitative samples ({qual_variant}, greedy)", ""]
    for i in picked:
        ref = tok.decode(store.get(i).tolist())
        lines += [f"## Sample {i} (n={len(store.get(i))})", "",
                  "| K | text |", "|---|------|",
                  f"| **original** | {esc(ref)} |"]
        for k in qks:
            if (qual_variant, k) in gen_store:
                lines.append(f"| {k} | {esc(gen_store[(qual_variant, k)][i])} |")
        lines.append("")
    lines += ["## Variant comparison at K=16", ""]
    for i in picked:
        ref = tok.decode(store.get(i).tolist())
        lines += [f"### Sample {i}", "", "| variant | text |", "|---|------|",
                  f"| original | {esc(ref)} |"]
        for v in others:
            if (v, 16) in gen_store:
                lines.append(f"| {v} | {esc(gen_store[(v, 16)][i])} |")
        lines.append("")
    open(path, "w").write("\n".join(lines))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
