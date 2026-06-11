"""Build wrap-up figures/CSVs from the partial 3000-step run.

Inputs:  results/train_{main,enc0,b2,b3}.csv  (val_loss trajectories)
         eval metrics for main + enc0 (captured from the eval console before the
         run was stopped to save cost; b1/b2/b3 eval + qualitative samples were
         not collected).
Outputs: results/training_curves.png, results/eval_curves.png,
         results/metrics_partial.csv
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), "..", "results")

# --- eval metrics captured from console (token_acc, bleu, chrf, semsim) ---
K = [1, 2, 4, 8, 16, 32, 64]
EVAL = {
    "main": {
        "acc": [0.0195, 0.0231, 0.0253, 0.0290, 0.0303, 0.0335, 0.0318],
        "bleu": [3.81, 3.88, 3.98, 4.10, 4.09, 4.39, 4.18],
        "chrf": [22.77, 23.38, 23.77, 23.96, 24.15, 24.46, 24.33],
        "sem": [0.3818, 0.3842, 0.3892, 0.3946, 0.3973, 0.4021, 0.3989],
    },
    "enc0": {
        "acc": [0.0205, 0.0242, 0.0279, 0.0341, 0.0412, 0.0477, 0.0485],
        "bleu": [4.37, 4.24, 4.63, 4.79, 5.01, 5.13, 5.18],
        "chrf": [23.37, 24.04, 24.40, 24.81, 25.07, 25.15, 24.94],
        "sem": [0.3450, 0.3461, 0.3463, 0.3531, 0.3573, 0.3589, 0.3588],
    },
}
LABEL = {"main": "main (DCT, L=4)", "enc0": "enc0 (DCT, L=0)",
         "b2": "B2 prefix-trunc", "b3": "B3 random-K"}


def write_partial_csv():
    path = os.path.join(RES, "metrics_partial.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "K", "token_acc", "bleu", "chrf", "semsim"])
        for v in ("main", "enc0"):
            for i, k in enumerate(K):
                w.writerow([v, k, EVAL[v]["acc"][i], EVAL[v]["bleu"][i],
                            EVAL[v]["chrf"][i], EVAL[v]["sem"][i]])
    print("wrote", path)


def read_val(v):
    steps, vals = [], []
    with open(os.path.join(RES, f"train_{v}.csv")) as f:
        for row in csv.DictReader(f):
            if row["val_loss"]:
                steps.append(int(row["step"]))
                vals.append(float(row["val_loss"]))
    return steps, vals


def training_curves():
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for v in ("main", "enc0", "b2", "b3"):
        s, vl = read_val(v)
        ax.plot(s, vl, marker="o", label=f"{LABEL[v]} (final {vl[-1]:.3f})")
    ax.set_xlabel("training step")
    ax.set_ylabel("validation loss (CE, log-uniform K)")
    ax.set_title("Validation loss by variant (3000-step budget run)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(RES, "training_curves.png")
    fig.savefig(p, dpi=150)
    print("wrote", p)


def eval_curves():
    panels = [("acc", "Token accuracy"), ("chrf", "chrF"),
              ("sem", "Semantic cosine sim")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (key, title) in zip(axes, panels):
        for v in ("main", "enc0"):
            ax.plot(K, EVAL[v][key], marker="o", label=LABEL[v])
        ax.set_xscale("log", base=2)
        ax.set_xlabel("K (kept coefficients)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9)
    fig.suptitle("Reconstruction quality vs K — main & enc0 (3000-step run, test=2000)")
    fig.tight_layout()
    p = os.path.join(RES, "eval_curves.png")
    fig.savefig(p, dpi=150)
    print("wrote", p)


if __name__ == "__main__":
    write_partial_csv()
    training_curves()
    eval_curves()
