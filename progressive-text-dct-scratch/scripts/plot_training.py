"""Plot val-loss training curves for the four variants -> results/training_curves.png"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), "..", "results")
LABEL = {"main": "main (DCT, learned E)", "b2": "B2 frozen-random E",
         "b3": "B3 random-K coeffs", "b4": "B4 prefix truncation"}

fig, ax = plt.subplots(figsize=(7.5, 4.8))
for v in ("main", "b2", "b3", "b4"):
    steps, vals = [], []
    with open(os.path.join(RES, f"train_{v}.csv")) as f:
        for row in csv.DictReader(f):
            if row["val_loss"]:
                steps.append(int(row["step"]))
                vals.append(float(row["val_loss"]))
    ax.plot(steps, vals, label=f"{LABEL[v]} (final {vals[-1]:.3f})", lw=1.2)
ax.set_xlabel("training step")
ax.set_ylabel("validation loss (CE, log-uniform K)")
ax.set_title("Validation loss, 100k steps x 4 variants (from-scratch)")
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(RES, "training_curves.png"), dpi=150)
print("saved", os.path.join(RES, "training_curves.png"))
