"""Train the progressive-DCT decoder (LoRA + projection/index/length embeddings).

Variants:
  main : encoder_layer=cfg (default 4), DCT, first-K coefficients
  enc0 : encoder_layer=0,               DCT, first-K coefficients
  b2   : encoder_layer=cfg, truncation  (hidden states of first K tokens)
  b3   : encoder_layer=cfg, DCT, random-K coefficient indices

Usage: python src/train.py --config config.yaml --variant main [--sanity]
"""

import argparse
import csv
import json
import math
import os
import random
import time

import torch
import yaml

from data import ChunkBatcher, attach_k, prepare_chunks
from model import (Compressor, DCTBank, DecoderExtras, PartialEncoder,
                   apply_lora, build_decoder_batch, build_prefixes,
                   compute_h_scale, encode_hiddens, get_backbone,
                   greedy_generate, load_base)

VARIANTS = {
    "main": {"mode": "dct"},
    "enc0": {"mode": "dct", "encoder_layer": 0},
    "b2": {"mode": "trunc"},
    "b3": {"mode": "dct"},  # random idx handled by the batcher
}


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_lambda(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def loss_on_items(lm, items, hiddens, compressor, extras, embed_tokens, device,
                  autocast_dtype):
    inputs, mask, labels = build_decoder_batch(
        items, hiddens, compressor, extras, embed_tokens, device)
    with torch.autocast("cuda", dtype=autocast_dtype):
        out = lm(inputs_embeds=inputs, attention_mask=mask, labels=labels)
    return out.loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--variant", default="main", choices=list(VARIANTS))
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sanity", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    vcfg = VARIANTS[args.variant]
    enc_layer = vcfg.get("encoder_layer", cfg["model"]["encoder_layer"])
    mode = vcfg["mode"]
    out_dir = args.out or os.path.join(
        "runs", ("sanity_" if args.sanity else "") + args.variant)
    os.makedirs(out_dir, exist_ok=True)
    set_seed(cfg["seed"])
    torch.set_float32_matmul_precision("high")
    device = "cuda"
    autocast_dtype = torch.bfloat16
    tcfg = dict(cfg["train"])
    if args.sanity:
        tcfg.update(cfg["sanity"])
    steps = args.steps or tcfg["steps"]
    micro_bs = tcfg["micro_batch_size"]
    accum = tcfg.get("grad_accum", 1)
    k_max = cfg["compress"]["k_max"]

    tok, lm_base = load_base(cfg, device)
    pad_id = tok.pad_token_id or tok.eos_token_id
    chunks = prepare_chunks(cfg, tok)
    lm = apply_lora(lm_base, cfg)
    for p in lm.parameters():  # fp32 master weights for LoRA params
        if p.requires_grad:
            p.data = p.data.float()
    backbone = get_backbone(lm)
    embed_tokens = backbone.embed_tokens
    d_model = lm.config.hidden_size
    extras = DecoderExtras(d_model, d_model, cfg["data"]["n_max"]).to(device)
    encoder = PartialEncoder(backbone, enc_layer)

    train_chunks = chunks["train"]
    if args.sanity:
        train_chunks = train_chunks[: cfg["sanity"]["num_samples"]]

    with lm.disable_adapter():
        h_scale = compute_h_scale(
            encoder, chunks["train"][: tcfg["calib_samples"]], device,
            pad_id=pad_id, autocast_dtype=autocast_dtype)
    print(f"[{args.variant}] encoder_layer={enc_layer} mode={mode} "
          f"h_scale={h_scale:.4f} steps={steps}")

    compressor = Compressor(DCTBank(device), mode, h_scale)
    batcher = ChunkBatcher(train_chunks, micro_bs, k_max, cfg["seed"],
                           variant=args.variant,
                           fixed_k=k_max if args.sanity else None)

    # validation uses the same log-uniform K distribution as training (seeded)
    val_items = attach_k(chunks["validation"][: tcfg["val_samples"]],
                         random.Random(cfg["seed"] + 777), k_max, args.variant)

    trainable_extras = list(extras.parameters())
    lora_params = [p for p in lm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": trainable_extras, "lr": tcfg["lr_proj"]},
         {"params": lora_params, "lr": tcfg["lr_lora"]}],
        weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, tcfg["warmup_steps"], steps))

    log_path = os.path.join(out_dir, "train_log.csv")
    logf = open(log_path, "w", newline="")
    logw = csv.writer(logf)
    logw.writerow(["step", "train_loss", "val_loss", "lr", "sec_per_step"])

    lm.train()
    ema = None
    t0 = time.time()
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            items = batcher.next_batch()
            with lm.disable_adapter():
                hiddens = encode_hiddens(encoder, items, device, pad_id=pad_id,
                                         autocast_dtype=autocast_dtype)
            loss = loss_on_items(lm, items, hiddens, compressor, extras,
                                 embed_tokens, device, autocast_dtype)
            (loss / accum).backward()
        torch.nn.utils.clip_grad_norm_(
            trainable_extras + lora_params, tcfg["grad_clip"])
        opt.step()
        sched.step()
        lval = loss.item()
        ema = lval if ema is None else 0.98 * ema + 0.02 * lval

        if step % tcfg["log_every"] == 0 or step == 1:
            sps = (time.time() - t0) / step
            print(f"step {step}/{steps} loss {ema:.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{sps:.3f}s/step", flush=True)

        if step % tcfg["val_every"] == 0 or step == steps:
            lm.eval()
            vloss, nb = 0.0, 0
            with torch.no_grad():
                for s in range(0, len(val_items), micro_bs):
                    vitems = val_items[s : s + micro_bs]
                    with lm.disable_adapter():
                        vh = encode_hiddens(encoder, vitems, device, pad_id=pad_id,
                                            autocast_dtype=autocast_dtype)
                    vloss += loss_on_items(lm, vitems, vh, compressor, extras,
                                           embed_tokens, device,
                                           autocast_dtype).item()
                    nb += 1
            lm.train()
            sps = (time.time() - t0) / step
            logw.writerow([step, f"{ema:.5f}", f"{vloss / nb:.5f}",
                           f"{sched.get_last_lr()[0]:.6e}", f"{sps:.3f}"])
            logf.flush()
            print(f"  -> val_loss {vloss / nb:.4f}", flush=True)
        elif step % tcfg["log_every"] == 0:
            logw.writerow([step, f"{ema:.5f}", "",
                           f"{sched.get_last_lr()[0]:.6e}",
                           f"{(time.time() - t0) / step:.3f}"])
            logf.flush()
    logf.close()

    from peft import get_peft_model_state_dict

    ckpt = {
        "variant": args.variant,
        "encoder_layer": enc_layer,
        "mode": mode,
        "h_scale": h_scale,
        "extras": extras.state_dict(),
        "lora": get_peft_model_state_dict(lm),
        "config": cfg,
        "steps_trained": steps,
        "wall_sec": time.time() - t0,
    }
    torch.save(ckpt, os.path.join(out_dir, "ckpt.pt"))
    print(f"saved {out_dir}/ckpt.pt ({(time.time() - t0) / 60:.1f} min)")

    if args.sanity:
        # reconstruct the overfitted samples at K=k_max and measure token acc
        lm.eval()
        match, total, exact = 0, 0, 0
        for s in range(0, len(train_chunks), micro_bs):
            toks_b = train_chunks[s : s + micro_bs]
            items = [{"tokens": t, "n": len(t), "k": min(k_max, len(t)),
                      "idx": list(range(min(k_max, len(t))))} for t in toks_b]
            with lm.disable_adapter():
                hb = encode_hiddens(encoder, items, device, pad_id=pad_id,
                                    autocast_dtype=autocast_dtype)
            pe, pm = build_prefixes(items, hb, compressor, extras, device)
            gen = greedy_generate(lm, embed_tokens, pe, pm,
                                  [it["n"] for it in items])
            for g, it in zip(gen, items):
                m = sum(int(a == b) for a, b in zip(g, it["tokens"]))
                match += m
                total += it["n"]
                exact += int(m == it["n"])
        acc = match / total
        print(f"SANITY token_acc={acc:.4f} exact_match={exact}/{len(train_chunks)}")
        with open(os.path.join(out_dir, "sanity.json"), "w") as f:
            json.dump({"token_acc": acc, "exact": exact,
                       "n": len(train_chunks)}, f)
        if acc < 0.99:
            print("SANITY FAILED: token accuracy below 99%")
            raise SystemExit(1)
        print("SANITY PASSED")


if __name__ == "__main__":
    main()
