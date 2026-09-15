"""
Training loop for the char-level GPT in model.py.

    python src/train.py --num-steps 5000 --batch-size 64

Batches are random contiguous windows of block_size+1 tokens sampled from the
train tensor -- there is no epoch structure and no shuffled dataset object,
because for a single flat token stream "sample a random offset" is the whole
data loader.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str):
    """Sample batch_size random windows. Returns x, y of shape (B, block_size).

    y is x shifted one position left: the target at position i is the token that
    actually followed position i, so a single forward pass gives us block_size
    next-token predictions instead of one.
    """
    # High end is len - block_size so that i + block_size + 1 stays in range.
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    if device == "cuda":
        # pin_memory + non_blocking lets the H2D copy overlap with compute.
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, splits, batch_size, block_size, device, eval_iters, ctx):
    """Average loss over eval_iters batches per split, with dropout disabled.

    Averaging over several batches matters here: a single batch of random
    windows is noisy enough that step-to-step val loss would look like it is
    bouncing around when it is really just sampling variance.
    """
    out = {}
    model.eval()
    for name, data in splits.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, batch_size, block_size, device)
            with ctx:
                _, loss, _ = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


# ---------------------------------------------------------------------------
# lr schedule
# ---------------------------------------------------------------------------

def get_lr(step: int, args) -> float:
    """Linear warmup, then cosine decay to min_lr, then flat.

    Warmup exists because Adam's second-moment estimate is garbage for the first
    few dozen steps -- taking full-size steps on that estimate is what blows up
    early training. Cosine decay is the standard "big steps early, fine steps
    late" schedule.
    """
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    if step > args.num_steps:
        return args.min_lr
    progress = (step - args.warmup_steps) / max(1, args.num_steps - args.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))          # 1 -> 0
    return args.min_lr + coeff * (args.lr - args.min_lr)


def configure_optimizer(model, weight_decay: float, lr: float, betas):
    """AdamW with weight decay on matmul weights only.

    Biases and LayerNorm gains are 1-D and get no decay: shrinking them toward
    zero is not regularization, it just fights the normalization the layer
    exists to do.
    """
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    print(
        f"optimizer: {len(decay)} decayed tensors ({sum(p.numel() for p in decay):,} params), "
        f"{len(no_decay)} undecayed ({sum(p.numel() for p in no_decay):,} params)"
    )
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    # data / io
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--out-dir", default=str(ROOT / "checkpoints"))
    # model
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--n-heads", type=int, default=6)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--tie-weights", action="store_true")
    # optimization
    p.add_argument("--num-steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # eval / logging
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--log-interval", type=int, default=50)
    # misc
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--compile", action="store_true", help="torch.compile the model")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    # TF32 for the fp32 matmuls: on Ampere and later this is a large free speedup
    # and the precision loss is irrelevant for training at this scale.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_data = torch.load(data_dir / "train.pt")
    val_data = torch.load(data_dir / "val.pt")
    tok = CharTokenizer.load(data_dir / "tokenizer.json")
    splits = {"train": train_data, "val": val_data}
    print(f"train {len(train_data):,} tokens | val {len(val_data):,} tokens | vocab {tok.vocab_size}")

    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        block_size=args.block_size,
        dropout=args.dropout,
        tie_weights=args.tie_weights,
    )
    model = GPT(cfg).to(args.device)
    print(f"model: {model.num_params():,} params  ({cfg.n_layers}L {cfg.n_heads}H {cfg.d_model}d)")

    # bf16 autocast on CUDA. bf16 rather than fp16 because it has the same
    # exponent range as fp32, so there is no loss-scaling machinery to get wrong.
    use_amp = args.device == "cuda" and torch.cuda.is_bf16_supported()
    ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp
        else torch.autocast(device_type="cpu", enabled=False)
    )
    print(f"autocast bf16: {use_amp}")

    opt = configure_optimizer(model, args.weight_decay, args.lr, betas=(0.9, 0.95))

    if args.compile:
        print("compiling model (first step will be slow)...")
        model = torch.compile(model)

    best_val = float("inf")
    t0 = time.time()

    for step in range(args.num_steps + 1):
        lr = get_lr(step, args)
        for g in opt.param_groups:
            g["lr"] = lr

        # ---- eval + checkpoint ----
        if step % args.eval_interval == 0 or step == args.num_steps:
            losses = estimate_loss(
                model, splits, args.batch_size, args.block_size,
                args.device, args.eval_iters, ctx,
            )
            elapsed = time.time() - t0
            print(
                f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                f"| lr {lr:.2e} | {elapsed:.0f}s"
            )
            if losses["val"] < best_val:
                best_val = losses["val"]
                ckpt = {
                    "model": getattr(model, "_orig_mod", model).state_dict(),
                    "config": cfg.__dict__,
                    "step": step,
                    "val_loss": best_val,
                    "args": vars(args),
                }
                torch.save(ckpt, out_dir / "best.pt")
                print(f"    saved best.pt (val {best_val:.4f})")

        if step == args.num_steps:
            break

        # ---- train step ----
        x, y = get_batch(train_data, args.batch_size, args.block_size, args.device)
        with ctx:
            _, loss, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Clip by global norm: one pathological batch (a wall of emoji, say)
        # should not be allowed to take a giant step and wreck the weights.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step % args.log_interval == 0:
            print(f"  step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e}")

    print(f"\ndone in {time.time() - t0:.0f}s | best val loss {best_val:.4f}")
    print(f"checkpoint: {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
