"""
Train a GPT on token files made by prepare_data.py.

    python prepare_data.py data/names.txt                        # once: makes data/processed/names
    python train.py data/processed/names                         # tiny model, under a minute

    python train.py data/processed/fineweb-edu --preset 300M     # big run (days)
    python train.py data/processed/fineweb-edu --resume          # continue after a stop
    python train.py data/processed/names --steps 1500 --lr 5e-4  # override any preset setting

Saves two files in checkpoints/:
    <name>.pt        the best model so far (lowest validation loss): use this with sample.py
    <name>-last.pt   the latest state, including the optimizer: used by --resume
"""

import argparse
import math
import os
import time
from contextlib import nullcontext

import torch

from data import get_batch, load_meta
from model import GPT, GPTConfig, load_checkpoint, save_checkpoint

# Model sizes, plus training settings that suit each one.
# Tokens learned per step = batch_size x grad_accum x block_size.
PRESETS = {
    # ~0.8M parameters: the original names demo (character tokens)
    "tiny": dict(n_layer=4, n_head=4, n_embd=128, block_size=32, dropout=0.1,
                 batch_size=64, grad_accum=1, lr=1e-3, warmup_steps=100, steps=3000,
                 eval_interval=250, log_interval=0),
    # ~10M parameters: a bigger character model, e.g. for a psychology textbook
    "small": dict(n_layer=6, n_head=6, n_embd=384, block_size=256, dropout=0.2,
                  batch_size=64, grad_accum=1, lr=1e-3, warmup_steps=200, steps=5000,
                  eval_interval=250, log_interval=50),
    # GPT-2 tokens from here on, ~0.5M tokens per step like GPT-2 and GPT-3.
    # batch_size is set to fit a 12 GB GPU (measured on an RTX 3060). batch_size x grad_accum
    # is what matters for learning, so with more GPU memory you can double batch_size and halve
    # grad_accum (a bit faster); out of memory, do the opposite.
    "50M": dict(n_layer=8, n_head=8, n_embd=512, block_size=1024, dropout=0.0,                 # 7.1 GB
                batch_size=8, grad_accum=32, lr=1e-3, warmup_steps=300, steps=4000,        # ~1B tokens
                eval_interval=250, log_interval=10),
    "125M": dict(n_layer=12, n_head=12, n_embd=768, block_size=1024, dropout=0.0,              # 10.0 GB
                 batch_size=8, grad_accum=64, lr=6e-4, warmup_steps=700, steps=4800,        # ~2.5B tokens
                 eval_interval=250, log_interval=10),
    "300M": dict(n_layer=20, n_head=16, n_embd=1024, block_size=1024, dropout=0.0,             # 11.0 GB
                 batch_size=4, grad_accum=128, lr=3e-4, warmup_steps=700, steps=11500,      # ~6B tokens
                 eval_interval=250, log_interval=10),
    "500M": dict(n_layer=22, n_head=20, n_embd=1280, block_size=1024, dropout=0.0,             # 10.8 GB
                 batch_size=1, grad_accum=512, lr=2.5e-4, warmup_steps=1000, steps=19000,   # ~10B tokens
                 eval_interval=250, log_interval=10),
}
MODEL_KEYS = ("n_layer", "n_head", "n_embd", "block_size", "dropout")
EVAL_BATCHES = 50     # average the loss over this many batches when checking progress
WEIGHT_DECAY = 0.1    # gently pulls weights toward zero, a brake on overfitting
GRAD_CLIP = 1.0       # caps the size of any single update, so one bad batch can't wreck training


def get_lr(step, s):
    """Learning-rate schedule: ramp up for warmup_steps, then glide down a cosine curve to 10%."""
    max_lr, min_lr = s["lr"], s["lr"] / 10
    if step < s["warmup_steps"]:
        return max_lr * (step + 1) / s["warmup_steps"]
    progress = min(1.0, (step - s["warmup_steps"]) / max(1, s["steps"] - s["warmup_steps"]))
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (max_lr - min_lr)


def make_optimizer(model, lr, device):
    """AdamW, with weight decay on the big weight matrices only (not on biases and LayerNorm)."""
    params = list(model.parameters())
    groups = [
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": WEIGHT_DECAY},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), fused=(device == "cuda"))


@torch.no_grad()
def estimate_loss(model, bins, block_size, batch_size, device, autocast):
    """Average the loss over many batches, so one lucky or unlucky batch doesn't skew it."""
    model.eval()                    # switch off dropout while measuring
    out = {}
    for split, path in bins.items():
        losses = torch.zeros(EVAL_BATCHES)
        for k in range(EVAL_BATCHES):
            x, y = get_batch(path, block_size, batch_size, device)
            with autocast:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def fmt_count(n):
    """6_144_000 -> '6.1M'."""
    for unit, size in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= size:
            return f"{n / size:.1f}{unit}"
    return str(n)


def fmt_time(seconds):
    """93784 -> '1d 2h'."""
    m, h, d = seconds / 60, seconds / 3600, seconds / 86400
    if d >= 1:
        return f"{int(d)}d {int(h % 24)}h"
    if h >= 1:
        return f"{int(h)}h {int(m % 60)}m"
    return f"{int(m)}m {int(seconds % 60)}s"


def main():
    parser = argparse.ArgumentParser(description="Train a GPT on token files made by prepare_data.py.")
    parser.add_argument("data_dir", nargs="?", default="data/processed/names",
                        help="folder made by prepare_data.py (default: data/processed/names)")
    parser.add_argument("--preset", choices=PRESETS, help="model size + training settings (default: tiny)")
    parser.add_argument("--resume", action="store_true", help="continue from checkpoints/<name>-last.pt")
    parser.add_argument("--out", help="where to save the best model (default: checkpoints/<data folder name>.pt)")
    parser.add_argument("--compile", action="store_true",
                        help="speed up with torch.compile (on Windows this needs the triton-windows package)")
    # Every preset setting can be overridden, e.g. --steps 1500 or --batch-size 4
    for key, value in PRESETS["tiny"].items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(value), help=f"override the preset's {key}")
    args = parser.parse_args()

    meta = load_meta(args.data_dir)
    bins = {split: os.path.join(args.data_dir, f"{split}.bin") for split in ("train", "val")}
    name = os.path.basename(os.path.normpath(args.data_dir))
    best_path = args.out or os.path.join("checkpoints", f"{name}.pt")
    last_path = os.path.splitext(best_path)[0] + "-last.pt"
    os.makedirs(os.path.dirname(best_path) or ".", exist_ok=True)

    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True     # faster maths on modern NVIDIA GPUs
    torch.backends.cudnn.allow_tf32 = True
    # bf16 = 16-bit numbers for most of the maths: about half the memory and much faster, same results
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()

    overrides = {k: getattr(args, k) for k in PRESETS["tiny"] if getattr(args, k) is not None}
    if args.resume:
        # Rebuild the model and optimizer exactly as they were; overrides (e.g. more --steps) still apply
        model, ckpt = load_checkpoint(last_path, device)
        model.train()
        if ckpt["tokenizer"] != meta["tokenizer"]:
            raise SystemExit(f"{last_path} was trained with a different tokenizer than {args.data_dir}")
        if args.preset:
            print("note: --preset is ignored when resuming; the checkpoint's settings are used")
        s = {**ckpt["settings"], **overrides}
        optimizer = make_optimizer(model, s["lr"], device)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step, best_val_loss = ckpt["step"], ckpt["best_val_loss"]
        print(f"resuming from {last_path} at step {start_step}")
    else:
        s = {**PRESETS[args.preset or "tiny"], **overrides}
        vocab_size = meta["vocab_size"]
        if meta["tokenizer"]["type"] == "gpt2":
            vocab_size = math.ceil(vocab_size / 64) * 64   # 50,257 -> 50,304: GPUs run faster on round sizes
        config = GPTConfig(vocab_size=vocab_size, **{k: s[k] for k in MODEL_KEYS})
        model = GPT(config).to(device)
        optimizer = make_optimizer(model, s["lr"], device)
        start_step, best_val_loss = 0, float("inf")

    block_size = model.config.block_size
    for split in bins:
        if meta[f"{split}_tokens"] <= block_size + 1:
            raise SystemExit(f"{split}.bin has only {meta[f'{split}_tokens']} tokens, too few for block_size {block_size}")

    raw_model = model                 # the plain model, for saving (a compiled model has renamed weights)
    if args.compile:
        model = torch.compile(model)

    n_params = sum(p.numel() for p in raw_model.parameters())
    tokens_per_step = s["batch_size"] * s["grad_accum"] * block_size
    total_tokens = tokens_per_step * s["steps"]
    print(f"data: {args.data_dir} ({fmt_count(meta['train_tokens'])} train tokens, {meta['tokenizer']['type']} tokenizer)")
    print(f"model: {n_params / 1e6:.1f}M parameters on {device}{' (bf16)' if use_bf16 else ''}")
    print(f"training: {s['steps']:,} steps x {fmt_count(tokens_per_step)} tokens = {fmt_count(total_tokens)} tokens "
          f"(~{total_tokens / meta['train_tokens']:.1f} passes over the data)")

    run_start = t_last = time.time()
    steps_since_log = 0
    for step in range(start_step, s["steps"] + 1):
        lr = get_lr(step, s)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Every so often: measure progress and save.
        # The best model is kept separately, so overfitting later in training can't spoil it.
        if step % s["eval_interval"] == 0 or step == s["steps"]:
            losses = estimate_loss(model, bins, block_size, s["batch_size"], device, autocast)
            line = f"step {step}: train loss {losses['train']:.3f} | val loss {losses['val']:.3f}"
            info = dict(step=step, val_loss=losses["val"], dataset=args.data_dir, settings=s)
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(best_path, raw_model, meta["tokenizer"], **info)
                line += "  -> saved best"
            if step > start_step:
                save_checkpoint(last_path, raw_model, meta["tokenizer"], **info,
                                best_val_loss=best_val_loss, optimizer=optimizer.state_dict())
            print(line)
            t_last, steps_since_log = time.time(), 0     # don't count evaluation in the speed numbers

        if step == s["steps"]:
            break

        # One step = grad_accum small batches whose directions are added up, then one nudge.
        # This acts like one big batch that wouldn't fit in GPU memory all at once.
        loss_sum = 0.0
        for _ in range(s["grad_accum"]):
            x, y = get_batch(bins["train"], block_size, s["batch_size"], device)
            with autocast:
                _, loss = model(x, y)                  # 1. guess + 2. measure
            (loss / s["grad_accum"]).backward()        # 3. find the direction
            loss_sum += loss.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()                               # 4. nudge
        optimizer.zero_grad(set_to_none=True)
        steps_since_log += 1

        if s["log_interval"] and step % s["log_interval"] == 0:
            loss_now = loss_sum.item() / s["grad_accum"]   # .item() also waits for the GPU to finish
            sec_per_step = (time.time() - t_last) / steps_since_log
            t_last, steps_since_log = time.time(), 0
            left = sec_per_step * (s["steps"] - step - 1)
            mem = ""
            if device == "cuda":
                peak, total = torch.cuda.max_memory_allocated(), torch.cuda.get_device_properties(0).total_memory
                mem = f" | {peak / 1e9:.1f}GB GPU"
            print(f"step {step}: loss {loss_now:.3f} | lr {lr:.1e} | {sec_per_step:.2f}s/step | "
                  f"{fmt_count(tokens_per_step / sec_per_step)} tokens/s{mem} | ~{fmt_time(left)} left")
            if device == "cuda" and peak > 0.95 * total:
                # On Windows the driver quietly spills into system RAM instead of crashing: very slow
                print(f"  WARNING: {peak / 1e9:.1f}GB of {total / 1e9:.1f}GB GPU memory used. At the limit, "
                      f"Windows spills into system RAM and training gets much slower. "
                      f"Halve --batch-size and double --grad-accum.")

    print(f"\ndone in {fmt_time(time.time() - run_start)}. best val loss {best_val_loss:.3f}, saved to {best_path}")
    print(f"generate text with: python sample.py {best_path}")


if __name__ == "__main__":
    main()
