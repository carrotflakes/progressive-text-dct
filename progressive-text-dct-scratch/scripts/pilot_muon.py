"""Pilot: AdamW vs hybrid Muon+AdamW on the task2 main arch, 3000 steps each.

Identical seed/data order/schedule; reports train EMA, val loss and s/step.
Run: PYTHONPATH=src ../.venv/bin/python scripts/pilot_muon.py
"""

import math
import random
import sys
import time

import torch
import yaml

sys.path.insert(0, "src")
from data import ChunkStore, make_batch, prepare_chunks  # noqa: E402
from model import ScratchLM  # noqa: E402
from muon import Muon, split_params  # noqa: E402
from tokenizer import load_tokenizer  # noqa: E402

STEPS = 3000
WARMUP = 300
VAL_EVERY = 500


def lr_lambda(step):
    if step < WARMUP:
        return step / WARMUP
    p = (step - WARMUP) / (STEPS - WARMUP)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def run(arm, cfg, train_store, val_batches, device):
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    model = ScratchLM(cfg, device).to(device)
    tcfg = cfg["train"]
    if arm == "muon":
        mp, ap = split_params(model)
        print(f"  muon params: {sum(p.numel() for p in mp)/1e6:.1f}M, "
              f"adamw params: {sum(p.numel() for p in ap)/1e6:.1f}M")
        opts = [torch.optim.AdamW(ap, lr=tcfg["lr"], betas=(0.9, 0.95),
                                  weight_decay=tcfg["weight_decay"]),
                Muon(mp, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])]
    else:
        opts = [torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                                  betas=(0.9, 0.95),
                                  weight_decay=tcfg["weight_decay"])]
    scheds = [torch.optim.lr_scheduler.LambdaLR(o, lr_lambda) for o in opts]
    rng = random.Random(cfg["seed"] + 1)
    k_max = cfg["compress"]["k_max"]
    n_max = cfg["data"]["n_max"]
    bs = tcfg["micro_batch_size"]

    model.train()
    ema = None
    t0 = time.time()
    history = []
    for step in range(1, STEPS + 1):
        for o in opts:
            o.zero_grad(set_to_none=True)
        indices = [rng.randrange(len(train_store)) for _ in range(bs)]
        batch = make_batch(train_store, indices, rng, k_max, n_max, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(*batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        for o in opts:
            o.step()
        for s in scheds:
            s.step()
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step % VAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                vl = 0.0
                for vb in val_batches:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        vl += model(*vb).item()
                vl /= len(val_batches)
            model.train()
            history.append((step, ema, vl))
            print(f"  [{arm}] step {step} train_ema {ema:.4f} val {vl:.4f} "
                  f"({(time.time()-t0)/step:.3f}s/step)", flush=True)
    return history


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    device = "cuda"
    torch.set_float32_matmul_precision("high")
    tok = load_tokenizer(cfg)
    blobs = prepare_chunks(cfg, tok)
    train_store = ChunkStore(blobs["train"])
    val_store = ChunkStore(blobs["validation"])
    vrng = random.Random(cfg["seed"] + 777)
    bs = cfg["train"]["micro_batch_size"]
    val_idx = list(range(512))
    val_batches = [make_batch(val_store, val_idx[s : s + bs], vrng,
                              cfg["compress"]["k_max"], cfg["data"]["n_max"],
                              device=device)
                   for s in range(0, len(val_idx), bs)]
    results = {}
    for arm in ("adamw", "muon"):
        print(f"=== {arm} ===", flush=True)
        results[arm] = run(arm, cfg, train_store, val_batches, device)
    print("\n=== summary (val loss) ===")
    print("step | adamw | muon")
    for (s, _, va), (_, _, vm) in zip(results["adamw"], results["muon"]):
        print(f"{s:5d} | {va:.4f} | {vm:.4f}")


if __name__ == "__main__":
    main()
