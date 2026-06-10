"""Evaluate trained variants + the untrained B1 baseline on the test set.

For each K in eval.k_values measures: token accuracy (positional match of
greedy generation vs reference), exact sequence match, BLEU, chrF
(sacrebleu), and semantic cosine similarity (all-MiniLM-L6-v2).

Outputs: results/metrics.csv, results/curves.png, results/samples.md
Usage: python src/evaluate.py --config config.yaml [--variants main,enc0,b2,b3,b1]
"""

import argparse
import csv
import os
import random

import torch
import yaml

from data import fixed_eval_items, prepare_chunks
from model import (Compressor, DCTBank, DecoderExtras, PartialEncoder,
                   apply_lora, build_prefixes, encode_hiddens, get_backbone,
                   greedy_generate, load_base)


def batches(lst, bs):
    for s in range(0, len(lst), bs):
        yield lst[s : s + bs]


def token_metrics(gens, refs):
    match = total = exact = 0
    for g, r in zip(gens, refs):
        m = sum(int(a == b) for a, b in zip(g, r))
        match += m
        total += len(r)
        exact += int(m == len(r) and len(g) == len(r))
    return match / total, exact / len(refs)


def text_metrics(hyp_texts, ref_texts, semodel):
    import sacrebleu

    bleu = sacrebleu.corpus_bleu(hyp_texts, [ref_texts]).score
    chrf = sacrebleu.corpus_chrf(hyp_texts, [ref_texts]).score
    h = semodel.encode(hyp_texts, batch_size=256, convert_to_tensor=True,
                       normalize_embeddings=True, show_progress_bar=False)
    r = semodel.encode(ref_texts, batch_size=256, convert_to_tensor=True,
                       normalize_embeddings=True, show_progress_bar=False)
    semsim = (h * r).sum(-1).mean().item()
    return bleu, chrf, semsim


@torch.no_grad()
def eval_trained(lm, encoder, compressor, extras, embed_tokens, tok, items,
                 device, gen_bs, pad_id):
    gens = []
    for chunk in batches(items, gen_bs):
        with lm.disable_adapter():
            hid = encode_hiddens(encoder, chunk, device, pad_id=pad_id)
        pe, pm = build_prefixes(chunk, hid, compressor, extras, device)
        gens += greedy_generate(lm, embed_tokens, pe, pm,
                                [it["n"] for it in chunk])
    return gens


@torch.no_grad()
def eval_b1(embed_weight, dct_bank, items, device, pos_chunk=2048):
    """Untrained decode: zero-pad coefficients -> inverse DCT -> nearest
    neighbour token in embedding space (cosine)."""
    e = embed_weight.to(device=device, dtype=torch.float32)
    e_n = torch.nn.functional.normalize(e, dim=-1)
    gens = []
    for it in items:
        ids = torch.tensor(it["tokens"], device=device)
        h = embed_weight[ids].to(torch.float32)  # (n, d) raw embeddings
        c = dct_bank.get(it["n"])
        z = c @ h
        keep = torch.zeros_like(z)
        idx = torch.tensor(it["idx"], device=device)
        keep[idx] = z[idx]
        h_hat = c.T @ keep
        h_hat = torch.nn.functional.normalize(h_hat, dim=-1)
        sims = []
        for s in range(0, h_hat.shape[0], pos_chunk):
            sims.append((h_hat[s : s + pos_chunk] @ e_n.T).argmax(-1))
        gens.append(torch.cat(sims).tolist())
    return gens


def load_variant(lm, extras, ckpt_path, device):
    from peft import set_peft_model_state_dict

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    set_peft_model_state_dict(lm, ckpt["lora"])
    extras.load_state_dict(ckpt["extras"])
    return ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--variants", default="main,enc0,b2,b3,b1")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N test chunks (debug)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    os.makedirs(args.out, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    device = "cuda"
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    tok, lm_base = load_base(cfg, device)
    pad_id = tok.pad_token_id or tok.eos_token_id
    chunks = prepare_chunks(cfg, tok)
    test_chunks = chunks["test"]
    if args.limit:
        test_chunks = test_chunks[: args.limit]
    lm = apply_lora(lm_base, cfg)
    lm.eval()
    backbone = get_backbone(lm)
    embed_tokens = backbone.embed_tokens
    d_model = lm.config.hidden_size
    extras = DecoderExtras(d_model, d_model, cfg["data"]["n_max"]).to(device)
    dct_bank = DCTBank(device)
    k_values = cfg["eval"]["k_values"]
    gen_bs = cfg["eval"]["gen_batch_size"]

    from sentence_transformers import SentenceTransformer

    semodel = SentenceTransformer(cfg["eval"]["semantic_model"], device=device)
    ref_texts = [tok.decode(c) for c in test_chunks]

    rows = []
    gen_store = {}  # (variant, K) -> list of hyp texts
    for variant in args.variants.split(","):
        if variant == "b1":
            for k in k_values:
                items = fixed_eval_items(test_chunks, k, "b1", cfg["seed"])
                gens = eval_b1(embed_tokens.weight.data, dct_bank, items, device)
                acc, em = token_metrics(gens, test_chunks)
                hyps = [tok.decode(g) for g in gens]
                bleu, chrf, sem = text_metrics(hyps, ref_texts, semodel)
                rows.append(["b1", k, acc, em, bleu, chrf, sem])
                gen_store[("b1", k)] = hyps
                print(f"b1 K={k}: acc={acc:.4f} em={em:.4f} bleu={bleu:.2f} "
                      f"chrf={chrf:.2f} sem={sem:.4f}", flush=True)
            continue

        ckpt_path = os.path.join(args.runs, variant, "ckpt.pt")
        if not os.path.exists(ckpt_path):
            print(f"skip {variant}: no checkpoint at {ckpt_path}")
            continue
        ckpt = load_variant(lm, extras, ckpt_path, device)
        encoder = PartialEncoder(backbone, ckpt["encoder_layer"])
        compressor = Compressor(dct_bank, ckpt["mode"], ckpt["h_scale"])
        for k in k_values:
            items = fixed_eval_items(test_chunks, k, variant, cfg["seed"])
            gens = eval_trained(lm, encoder, compressor, extras, embed_tokens,
                                tok, items, device, gen_bs, pad_id)
            acc, em = token_metrics(gens, test_chunks)
            hyps = [tok.decode(g) for g in gens]
            bleu, chrf, sem = text_metrics(hyps, ref_texts, semodel)
            rows.append([variant, k, acc, em, bleu, chrf, sem])
            gen_store[(variant, k)] = hyps
            print(f"{variant} K={k}: acc={acc:.4f} em={em:.4f} bleu={bleu:.2f} "
                  f"chrf={chrf:.2f} sem={sem:.4f}", flush=True)

    with open(os.path.join(args.out, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "K", "token_acc", "exact_match", "bleu",
                    "chrf", "semsim"])
        w.writerows(rows)

    plot_curves(rows, os.path.join(args.out, "curves.png"))
    write_samples(gen_store, test_chunks, ref_texts, cfg,
                  os.path.join(args.out, "samples.md"))
    print("done")


def plot_curves(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [("token_acc", "Token accuracy"), ("chrf", "chrF"),
               ("semsim", "Semantic cosine sim")]
    cols = {"token_acc": 2, "chrf": 5, "semsim": 6}
    variants = sorted({r[0] for r in rows},
                      key=lambda v: ["main", "enc0", "b1", "b2", "b3"].index(v)
                      if v in ["main", "enc0", "b1", "b2", "b3"] else 9)
    labels = {"main": "main (DCT, L=4)", "enc0": "DCT, L=0 (trained)",
              "b1": "B1 untrained NN (L=0)", "b2": "B2 prefix-truncation",
              "b3": "B3 random-K coeffs"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (mkey, title) in zip(axes, metrics):
        for v in variants:
            pts = sorted([(r[1], r[cols[mkey]]) for r in rows if r[0] == v])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                    label=labels.get(v, v))
        ax.set_xscale("log", base=2)
        ax.set_xlabel("K (kept coefficients)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Reconstruction quality vs K")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


def write_samples(gen_store, test_chunks, ref_texts, cfg, path):
    qks = cfg["eval"]["qualitative_ks"]
    qn = cfg["eval"]["qualitative_n"]
    # pick the first qn test chunks of readable length
    picked = [i for i, c in enumerate(test_chunks) if 40 <= len(c) <= 90][:qn]
    lines = ["# Qualitative samples (main method, greedy decoding)", ""]
    for i in picked:
        lines.append(f"## Sample {i} (n={len(test_chunks[i])} tokens)")
        lines.append("")
        lines.append("| K | text |")
        lines.append("|---|------|")
        esc = lambda t: t.replace("|", "\\|").replace("\n", " ⏎ ")  # noqa: E731
        lines.append(f"| **original** | {esc(ref_texts[i])} |")
        for k in qks:
            if ("main", k) in gen_store:
                lines.append(f"| {k} | {esc(gen_store[('main', k)][i])} |")
        lines.append("")
    lines.append("## Baseline reconstructions at K=16 (same samples)")
    lines.append("")
    for i in picked:
        lines.append(f"### Sample {i}")
        lines.append("")
        lines.append("| variant | text |")
        lines.append("|---------|------|")
        esc = lambda t: t.replace("|", "\\|").replace("\n", " ⏎ ")  # noqa: E731
        lines.append(f"| original | {esc(ref_texts[i])} |")
        for v in ["enc0", "b1", "b2", "b3"]:
            if (v, 16) in gen_store:
                lines.append(f"| {v} | {esc(gen_store[(v, 16)][i])} |")
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
