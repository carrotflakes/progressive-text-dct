"""Chunk wikitext-103 into 32..128-token chunks with the scratch BPE tokenizer.

Train uses the FULL train split (per spec); val/test come from the
validation/test splits. Chunks are stored as a flat int16 token array +
offsets for memory efficiency (~1.6M chunks).
"""

import math
import os
import random

import numpy as np
import torch


def prepare_chunks(cfg, tokenizer):
    dcfg = cfg["data"]
    os.makedirs(dcfg["cache_dir"], exist_ok=True)
    cache = os.path.join(dcfg["cache_dir"], "chunks_bpe16k.pt")
    if os.path.exists(cache):
        return torch.load(cache, weights_only=False)

    from datasets import load_dataset

    ds = load_dataset(dcfg["dataset"], dcfg["dataset_config"])
    out = {}
    for split, num in (("train", dcfg["num_train_chunks"]),
                       ("validation", dcfg["num_val"]),
                       ("test", dcfg["num_test"])):
        rng = random.Random(cfg["seed"] + hash(split) % 1000)
        d = ds[split]
        flat, offsets = [], [0]
        target = rng.randint(dcfg["n_min"], dcfg["n_max"])
        buf = []
        done = False
        for s in range(0, len(d), 20000):
            text = "".join(d[s : s + 20000]["text"])
            if not text:
                continue
            for tok in tokenizer.encode(text).ids:
                buf.append(tok)
                if len(buf) == target:
                    flat.extend(buf)
                    offsets.append(len(flat))
                    buf = []
                    target = rng.randint(dcfg["n_min"], dcfg["n_max"])
                    if num > 0 and len(offsets) - 1 >= num:
                        done = True
                        break
            if done:
                break
        out[split] = {
            "tokens": np.asarray(flat, dtype=np.int16),
            "offsets": np.asarray(offsets, dtype=np.int64),
        }
        n_chunks = len(offsets) - 1
        print(f"{split}: {n_chunks} chunks, avg len {len(flat) / max(1, n_chunks):.1f}")
    torch.save(out, cache)
    return out


class ChunkStore:
    """Random access view over the flat chunk arrays."""

    def __init__(self, blob):
        self.tokens = blob["tokens"]
        self.offsets = blob["offsets"]

    def __len__(self):
        return len(self.offsets) - 1

    def get(self, i):
        return self.tokens[self.offsets[i] : self.offsets[i + 1]].astype(np.int64)


def sample_k(rng, k_max):
    k = round(math.exp(rng.uniform(0.0, math.log(k_max))))
    return max(1, min(k_max, k))


def make_batch(store, indices, rng, k_max, n_max, variant="main", fixed_k=None,
               device="cuda"):
    """Vectorized batch: ids (B, n_max) 0-padded, lens, idx (B, k_pad), idx_valid.

    variant 'b3' draws K random coefficient indices (sorted); others take 0..K-1.
    """
    B = len(indices)
    lens = np.empty(B, dtype=np.int64)
    ks = np.empty(B, dtype=np.int64)
    ids = np.zeros((B, n_max), dtype=np.int64)
    for j, i in enumerate(indices):
        toks = store.get(i)
        n = len(toks)
        ids[j, :n] = toks
        lens[j] = n
        k = fixed_k if fixed_k is not None else sample_k(rng, k_max)
        ks[j] = min(k, n)
    k_pad = int(ks.max())
    idx = np.zeros((B, k_pad), dtype=np.int64)
    valid = np.zeros((B, k_pad), dtype=bool)
    for j in range(B):
        kj = ks[j]
        if variant == "b3":
            sel = sorted(rng.sample(range(int(lens[j])), int(kj)))
        else:
            sel = range(int(kj))
        idx[j, :kj] = np.fromiter(sel, dtype=np.int64, count=int(kj))
        valid[j, :kj] = True
    to = lambda a, dt: torch.from_numpy(a).to(device=device, dtype=dt)  # noqa: E731
    return (to(ids, torch.long), to(lens, torch.long),
            to(idx, torch.long), to(valid, torch.bool))


def eval_batch(store, indices, k, variant, seed, n_max, device="cuda"):
    """Deterministic fixed-K batch (B3 indices seeded per sample index)."""
    B = len(indices)
    lens = np.empty(B, dtype=np.int64)
    ids = np.zeros((B, n_max), dtype=np.int64)
    ks = np.empty(B, dtype=np.int64)
    for j, i in enumerate(indices):
        toks = store.get(i)
        ids[j, : len(toks)] = toks
        lens[j] = len(toks)
        ks[j] = min(k, len(toks))
    k_pad = int(ks.max())
    idx = np.zeros((B, k_pad), dtype=np.int64)
    valid = np.zeros((B, k_pad), dtype=bool)
    for j, i in enumerate(indices):
        kj = ks[j]
        if variant == "b3":
            rng = random.Random(seed * 1000003 + int(i))
            sel = sorted(rng.sample(range(int(lens[j])), int(kj)))
            idx[j, :kj] = np.fromiter(sel, dtype=np.int64, count=int(kj))
        else:
            idx[j, :kj] = np.arange(kj)
        valid[j, :kj] = True
    to = lambda a, dt: torch.from_numpy(a).to(device=device, dtype=dt)  # noqa: E731
    return (to(ids, torch.long), to(lens, torch.long),
            to(idx, torch.long), to(valid, torch.bool))
