"""Data preparation: chunk wikitext-103-raw-v1 into token chunks of 32..128 tokens.

Train chunks come from the train split; val/test chunks come from the
validation/test splits respectively so that evaluation text is never seen
during training.
"""

import os
import random

import torch


def _chunk_token_stream(token_iter, rng, n_min, n_max, num_chunks):
    """Cut a stream of token ids into chunks with lengths ~ U[n_min, n_max]."""
    chunks = []
    buf = []
    target = rng.randint(n_min, n_max)
    for tok in token_iter:
        buf.append(tok)
        if len(buf) == target:
            chunks.append(buf)
            if len(chunks) >= num_chunks:
                return chunks
            buf = []
            target = rng.randint(n_min, n_max)
    return chunks


def _tokens_from_split(dataset, tokenizer, batch_rows=2000):
    """Yield token ids from a wikitext split, tokenizing in row batches."""
    n = len(dataset)
    for start in range(0, n, batch_rows):
        rows = dataset[start : start + batch_rows]["text"]
        text = "".join(rows)
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        yield from ids


def prepare_chunks(cfg, tokenizer):
    """Return dict split -> list[list[int]] of token chunks, cached on disk."""
    dcfg = cfg["data"]
    cache_dir = dcfg["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)
    model_tag = cfg["model"]["base_model"].replace("/", "__")
    cache_path = os.path.join(cache_dir, f"chunks_{model_tag}.pt")
    if os.path.exists(cache_path):
        return torch.load(cache_path)

    from datasets import load_dataset

    ds = load_dataset(dcfg["dataset"], dcfg["dataset_config"])
    wanted = {
        "train": dcfg["num_train_chunks"],
        "validation": dcfg["num_val"],
        "test": dcfg["num_test"],
    }
    out = {}
    for split, num in wanted.items():
        rng = random.Random(cfg["seed"] + hash(split) % 1000)
        stream = _tokens_from_split(ds[split], tokenizer)
        chunks = _chunk_token_stream(stream, rng, dcfg["n_min"], dcfg["n_max"], num)
        if len(chunks) < num:
            print(f"WARNING: split {split} produced only {len(chunks)}/{num} chunks")
        out[split] = chunks
        print(f"{split}: {len(chunks)} chunks, "
              f"avg len {sum(map(len, chunks)) / len(chunks):.1f}")
    torch.save(out, cache_path)
    return out


def sample_k(rng, k_max):
    """K = round(exp(uniform(log 1, log k_max))), clamped to [1, k_max]."""
    import math

    k = round(math.exp(rng.uniform(0.0, math.log(k_max))))
    return max(1, min(k_max, k))


def attach_k(tokens_list, rng, k_max, variant="main", fixed_k=None):
    """Attach per-sample K (log-uniform unless fixed) and coefficient indices."""
    items = []
    for toks in tokens_list:
        n = len(toks)
        k = fixed_k if fixed_k is not None else sample_k(rng, k_max)
        k_eff = min(k, n)
        if variant == "b3":
            # random K coefficient indices out of the n available, sorted
            idx = sorted(rng.sample(range(n), k_eff))
        else:
            idx = list(range(k_eff))
        items.append({"tokens": toks, "n": n, "k": k_eff, "idx": idx})
    return items


class ChunkBatcher:
    """Yields batches of token chunks with per-sample K (and B3 coefficient
    indices) sampled from a seeded RNG. Single-process, infinite iterator."""

    def __init__(self, chunks, batch_size, k_max, seed, variant="main",
                 fixed_k=None):
        self.chunks = chunks
        self.batch_size = batch_size
        self.k_max = k_max
        self.variant = variant
        self.fixed_k = fixed_k
        self.rng = random.Random(seed)
        self.order = list(range(len(chunks)))
        self.pos = len(chunks)  # force shuffle on first batch

    def next_batch(self):
        batch = []
        for _ in range(self.batch_size):
            if self.pos >= len(self.order):
                self.rng.shuffle(self.order)
                self.pos = 0
            batch.append(self.chunks[self.order[self.pos]])
            self.pos += 1
        return attach_k(batch, self.rng, self.k_max, self.variant, self.fixed_k)


def fixed_eval_items(chunks, k, variant, seed):
    """Deterministic eval items at a fixed K (B3 indices seeded per sample)."""
    items = []
    for i, toks in enumerate(chunks):
        n = len(toks)
        k_eff = min(k, n)
        if variant == "b3":
            rng = random.Random(seed * 1000003 + i)
            idx = sorted(rng.sample(range(n), k_eff))
        else:
            idx = list(range(k_eff))
        items.append({"tokens": toks, "n": n, "k": k_eff, "idx": idx})
    return items
