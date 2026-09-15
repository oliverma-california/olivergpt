"""
Baseline autoregressive generation -- intentionally the slow version.

Every step re-runs a full forward pass over the entire sequence so far and then
throws away all but the last position's logits. That is O(T^2) work per step and
O(T^3) over a generation of length T, and recomputing K/V for tokens that have
not changed is exactly the waste the Phase 2 KV cache removes.

    python src/generate_naive.py --prompt "i think" --max-new-tokens 300
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model import GPT, GPTConfig
from tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent.parent


def load_checkpoint(path: str | Path, device: str):
    """Rebuild the model + tokenizer from a training checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()  # disables dropout -- essential for reproducible generation
    return model, cfg, ckpt


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    greedy: bool = False,
) -> torch.Tensor:
    """logits (B, vocab) -> next token ids (B, 1).

    greedy=True is argmax, which is what the correctness tests use: it removes
    all sampling randomness, so any difference between two implementations is a
    real numerical difference and not just a different draw from the RNG.
    """
    if greedy:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / max(temperature, 1e-8)
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        # Everything below the k-th largest logit becomes -inf, so softmax gives
        # it exactly zero probability. Keeps the long tail of implausible
        # characters from ever being drawn.
        kth = torch.topk(logits, k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_naive(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    greedy: bool = False,
    return_logits: bool = False,
):
    """idx: (B, T) prompt token ids. Returns (B, T + max_new_tokens).

    If return_logits, also returns the list of per-step last-position logits,
    which verify_correctness.py compares against the cached implementation.
    """
    block_size = model.config.block_size
    step_logits = []

    for _ in range(max_new_tokens):
        # Crop to the last block_size tokens: learned positional embeddings only
        # exist for indices < block_size, so anything older is simply unusable.
        idx_cond = idx[:, -block_size:]
        logits, _, _ = model(idx_cond)          # (B, T, vocab) -- the wasteful part
        last = logits[:, -1, :]              # (B, vocab)
        if return_logits:
            step_logits.append(last.clone())
        next_id = sample_from_logits(last, temperature, top_k, greedy)
        idx = torch.cat([idx, next_id], dim=1)

    return (idx, step_logits) if return_logits else idx


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "best.pt"))
    p.add_argument("--tokenizer", default=str(ROOT / "data" / "tokenizer.json"))
    p.add_argument("--prompt", default="\n")
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    model, cfg, ckpt = load_checkpoint(args.ckpt, args.device)
    tok = CharTokenizer.load(args.tokenizer)
    print(
        f"loaded {args.ckpt} (step {ckpt['step']}, val loss {ckpt['val_loss']:.4f}), "
        f"{model.num_params():,} params on {args.device}"
    )

    ids = tok.encode(args.prompt) or [tok.stoi.get("\n", 0)]
    idx = torch.tensor([ids], dtype=torch.long, device=args.device)

    if args.device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = generate_naive(
        model, idx, args.max_new_tokens,
        temperature=args.temperature, top_k=args.top_k, greedy=args.greedy,
    )
    if args.device == "cuda":
        # Without this the timer would stop before the GPU has actually finished:
        # CUDA launches are asynchronous.
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    print("-" * 70)
    print(tok.decode(out[0].tolist()))
    print("-" * 70)
    print(
        f"naive: {args.max_new_tokens} tokens in {dt:.2f}s "
        f"= {args.max_new_tokens / dt:.1f} tok/s"
    )


if __name__ == "__main__":
    main()
