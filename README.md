# Progressive Text Compression via Sequence-direction DCT

Progressive-JPEG-style text compression: encode text into an LM's hidden-state
sequence, apply a **DCT along the sequence axis**, keep only the **first K
coefficients**, and train (LoRA) a decoder to reconstruct the text from the
truncated coefficients. Increasing K should monotonically improve fidelity.

See [task.md](task.md) for the full spec and [report.md](report.md) for results.

## Layout

```
config.yaml          all hyperparameters (seed, model, data, train, eval)
src/data.py          chunk wikitext-103, log-uniform K sampling, batching
src/model.py         orthonormal DCT, frozen partial encoder, decoder, generation
src/train.py         train one variant (main / enc0 / b2 / b3), + sanity mode
src/evaluate.py      test-set metrics, B1 baseline, curves.png, samples.md
src/test_dct.py      DCT unit tests (vs scipy)
scripts/setup.sh     fresh-machine setup (venv + deps + model/data cache)
scripts/run_sanity.sh   Phase 1 sanity check (overfit 100 samples)
scripts/run_train.sh    train the 4 variants
scripts/run_eval.sh     evaluate
scripts/run_all.sh      train all 4 variants then evaluate (Phase 2+3)
```

## Variants

| variant | encoder_layer | compression          | tests   |
|---------|---------------|----------------------|---------|
| main    | 4             | DCT, first-K coeffs  | H1, H2  |
| enc0    | 0             | DCT, first-K coeffs  | H3 (B1 context) |
| b2      | 4             | first-K token states | H4      |
| b3      | 4             | random-K coeffs      | order   |
| B1      | 0 (untrained) | inverse-DCT + NN tok | H3      |

## Running on RunPod (RTX 4090, 24GB) — recommended

```bash
# in a CUDA-enabled PyTorch pod
git clone https://github.com/carrotflakes/progressive-text-dct.git
cd progressive-text-dct
PYTHON=python3 bash scripts/setup.sh        # venv + deps + download model/data (~5 min)
bash scripts/run_sanity.sh                  # confirm token_acc ~1.0 (~15 min)
bash scripts/run_all.sh                     # train 4 variants + evaluate (~4-5 h)
# results land in results/ (metrics.csv, curves.png, samples.md)
git add -A results report.md && git commit -m results && git push   # save back
```

`config.yaml` defaults are tuned for the 4090: micro_batch_size=16, grad_accum=2
(effective batch 32). Memory is dominated by the 152k-vocab logits, so batch 16
already uses ~16GB. On a bigger card you can raise `micro_batch_size` (and drop
grad_accum). Override the step count without editing the config:

```bash
bash scripts/run_all.sh 2000    # cheaper: 2000 steps per variant
```

### Unattended run with auto-stop (don't pay for idle time)

RunPod pods have no dashboard "idle auto-stop", but a pod can stop *itself* when
the job finishes (`runpodctl` + `$RUNPOD_POD_ID` are preinstalled). Use a PAT so
results are pushed back before the pod terminates:

```bash
export GITHUB_TOKEN=ghp_xxx     # PAT with 'repo' (contents:write) scope
export STOP_MODE=remove         # remove = terminate (no billing) | stop = keep disk
bash scripts/run_unattended.sh  # setup -> sanity -> train+eval -> push -> self-stop
```

`set -e` aborts before the self-stop on any failure, so a broken run leaves the
pod alive for debugging; a successful run pushes results, then removes the pod.

### Rough time/quality guidance (per-variant `steps`, ×4 variants)

The default (3000) is budget-leaning. Hypotheses H1–H4 are checkable even at the
low end; higher steps mainly sharpen near-perfect reconstruction at large K.

| steps/variant | eff. epochs | ~time/run (4090) | use case                 |
|---------------|-------------|------------------|--------------------------|
| 2000          | ~0.3        | ~40 min          | cheapest, rough curves   |
| 3000 (default)| ~0.5        | ~1 h             | budget hypothesis check  |
| 6000          | ~1.0        | ~2 h             | solid reconstruction     |
| 12000+        | ~1.9        | ~4 h             | best fidelity at high K  |

## Local development (smaller GPU)

Tested on an RTX 4070 (16GB): set `micro_batch_size: 16` in config.yaml. The
sanity check and a scaled-down train both run there; the full A6000 batch will
OOM on 16GB.

## Determinism

Seed is fixed (42) for data chunking, K sampling, and init. The DCT is
orthonormal (`norm='ortho'`), so coefficient scale is K-independent.
