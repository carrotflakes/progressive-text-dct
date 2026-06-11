"""Train the scratch prefix-LM decoder end-to-end through the DCT bottleneck.

Variants:
  main : E trainable, DCT first-K coefficients
  b2   : E frozen at random init (decoder still trained)   -> H3 contrast
  b3   : E trainable, random-K coefficient indices         -> H4 (order)
  b4   : E trainable, first-K token embeddings (truncation)-> H4 (prefix)

Checkpoints every train.ckpt_every_sec (resumable with --resume).
Usage: python src/train.py --config config.yaml --variant main [--sanity]
"""

import argparse
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
import yaml

from data import ChunkStore, eval_batch, make_batch, prepare_chunks
from tokenizer import load_tokenizer
from model import ScratchLM

VARIANTS = {"main": "dct", "b2": "dct", "b3": "dct", "b4": "trunc"}


def lr_lambda(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--variant", default="main", choices=list(VARIANTS))
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    mode = VARIANTS[args.variant]
    out_dir = os.path.join("runs", ("sanity_" if args.sanity else "") + args.variant)
    os.makedirs(out_dir, exist_ok=True)
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    device = "cuda"

    tcfg = dict(cfg["train"])
    if args.sanity:
        tcfg.update(cfg["sanity"])
    steps = args.steps or tcfg["steps"]
    bs = tcfg["micro_batch_size"]
    accum = tcfg.get("grad_accum", 1)
    k_max = cfg["compress"]["k_max"]
    n_max = cfg["data"]["n_max"]

    tok = load_tokenizer(cfg)
    blobs = prepare_chunks(cfg, tok)
    train_store = ChunkStore(blobs["train"])
    val_store = ChunkStore(blobs["validation"])
    n_train = len(train_store)
    if args.sanity:
        n_train = cfg["sanity"]["num_samples"]

    model = ScratchLM(cfg, device).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.variant}] mode={mode} params={n_params/1e6:.1f}M "
          f"steps={steps} eff_batch={bs*accum}", flush=True)
    if args.variant == "b2":
        model.enc_emb.weight.requires_grad_(False)   # frozen random E

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tcfg["lr"],
                            weight_decay=tcfg["weight_decay"], betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, tcfg["warmup_steps"], steps))

    start_step = 0
    rng = random.Random(seed + 1)
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    if args.resume and os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(sd["model"])
        opt.load_state_dict(sd["opt"])
        sched.load_state_dict(sd["sched"])
        start_step = sd["step"]
        rng.setstate(sd["rng"])
        print(f"resumed from step {start_step}", flush=True)

    log_path = os.path.join(out_dir, "train_log.csv")
    logf = open(log_path, "a" if start_step else "w", newline="")
    logw = csv.writer(logf)
    if not start_step:
        logw.writerow(["step", "train_loss", "val_loss", "lr", "sec_per_step"])

    # fixed validation batches (same log-uniform K distribution, seeded)
    vrng = random.Random(seed + 777)
    val_idx = list(range(min(tcfg["val_samples"], len(val_store))))
    val_batches = [make_batch(val_store, val_idx[s : s + bs], vrng, k_max, n_max,
                              variant=args.variant, device=device)
                   for s in range(0, len(val_idx), bs)]

    def save_ckpt(step):
        tmp = ckpt_path + ".tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step,
                    "rng": rng.getstate(), "cfg": cfg,
                    "variant": args.variant, "mode": mode}, tmp)
        os.replace(tmp, ckpt_path)

    model.train()
    ema = None
    t0 = time.time()
    last_ckpt = t0
    fixed_k = k_max if args.sanity else None
    for step in range(start_step + 1, steps + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            indices = [rng.randrange(n_train) for _ in range(bs)]
            ids, lens, idx, valid = make_batch(
                train_store, indices, rng, k_max, n_max,
                variant=args.variant, fixed_k=fixed_k, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(ids, lens, idx, valid, mode=mode)
            (loss / accum).backward()
        torch.nn.utils.clip_grad_norm_(params, tcfg["grad_clip"])
        opt.step()
        sched.step()
        lval = loss.item()
        ema = lval if ema is None else 0.98 * ema + 0.02 * lval

        if step % tcfg["log_every"] == 0 or step == 1:
            sps = (time.time() - t0) / (step - start_step)
            print(f"step {step}/{steps} loss {ema:.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e} {sps:.3f}s/step", flush=True)

        if step % tcfg["val_every"] == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                vloss = 0.0
                for vb in val_batches:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        vloss += model(*vb, mode=mode).item()
                vloss /= len(val_batches)
            model.train()
            sps = (time.time() - t0) / (step - start_step)
            logw.writerow([step, f"{ema:.5f}", f"{vloss:.5f}",
                           f"{sched.get_last_lr()[0]:.6e}", f"{sps:.3f}"])
            logf.flush()
            print(f"  -> val_loss {vloss:.4f}", flush=True)
        elif step % tcfg["log_every"] == 0:
            logw.writerow([step, f"{ema:.5f}", "",
                           f"{sched.get_last_lr()[0]:.6e}",
                           f"{(time.time() - t0) / (step - start_step):.3f}"])
            logf.flush()

        if time.time() - last_ckpt > tcfg["ckpt_every_sec"]:
            save_ckpt(step)
            last_ckpt = time.time()
            print(f"  [ckpt @ step {step}]", flush=True)

    save_ckpt(steps)
    logf.close()
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {ckpt_path}", flush=True)

    if args.sanity:
        model.eval()
        match = total = exact = 0
        for s in range(0, n_train, bs):
            idxs = list(range(s, min(s + bs, n_train)))
            ids, lens, idx, valid = eval_batch(
                train_store, idxs, k_max, args.variant, seed, n_max, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                gen = model.generate(ids, lens, idx, valid, mode=mode)
            for j in range(len(idxs)):
                n = int(lens[j])
                m = int((gen[j, :n] == ids[j, :n]).sum())
                match += m
                total += n
                exact += int(m == n)
        acc = match / total
        print(f"SANITY token_acc={acc:.4f} exact={exact}/{n_train}")
        json.dump({"token_acc": acc, "exact": exact, "n": n_train},
                  open(os.path.join(out_dir, "sanity.json"), "w"))
        if acc < 0.99:
            print("SANITY FAILED (<99%)")
            raise SystemExit(1)
        print("SANITY PASSED")


if __name__ == "__main__":
    main()
