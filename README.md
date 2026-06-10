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

## Running on RunPod (A6000, 48GB) — recommended

```bash
# in a CUDA-enabled PyTorch pod, repo checked out at /workspace/progressive-text-dct
cd /workspace/progressive-text-dct
PYTHON=python3 bash scripts/setup.sh        # venv + deps + download model/data (~5 min)
bash scripts/run_sanity.sh                  # confirm token_acc ~1.0 (~15 min)
bash scripts/run_all.sh                     # train 4 variants + evaluate
# results land in results/ (metrics.csv, curves.png, samples.md)
```

`config.yaml` defaults are tuned for 48GB (micro_batch_size=48, grad_accum=1).
Lower `micro_batch_size` for smaller cards. Override step count without editing
the config:

```bash
bash scripts/run_all.sh 4000    # 4000 steps per variant instead of the default
```

### Rough time/quality guidance

`steps` is per-variant; there are 4 variants. Pick based on your budget — the
hypotheses (H1–H4) are checkable even at the lower end; higher steps mainly
sharpen near-perfect reconstruction at large K.

| steps/variant | eff. epochs | use case                 |
|---------------|-------------|--------------------------|
| 2500          | ~0.6        | fast smoke of all curves |
| 4000          | ~1.0        | solid hypothesis check   |
| 8000 (default)| ~1.9        | good reconstruction      |
| 15000+        | ~3.6        | best fidelity at high K  |

## Local development (smaller GPU)

Tested on an RTX 4070 (16GB): set `micro_batch_size: 16` in config.yaml. The
sanity check and a scaled-down train both run there; the full A6000 batch will
OOM on 16GB.

## Determinism

Seed is fixed (42) for data chunking, K sampling, and init. The DCT is
orthonormal (`norm='ortho'`), so coefficient scale is K-independent.
