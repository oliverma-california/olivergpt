"""
KV-cached generation.

The cache is a plain Python list of per-layer (k, v) tuples:

    past_kvs[layer] = (k, v)   each of shape (B, n_heads, S, head_dim)

A list of tuples rather than one stacked (n_layers, 2, B, nh, S, hd) tensor,
because every step appends S -> S+1 along the sequence dim. On a list, that is
n_layers independent `torch.cat`s; on a single stacked tensor it would be one
cat over a bigger buffer, which is only a win if you preallocate to max length
and write in place. Preallocation is the right answer for a serving system --
it also makes the Phase 3 cache-trim a pointer move instead of a realloc -- but
it hardcodes a max length and obscures what the cache actually is, so the list
stays here.

Cache size is `2 * n_layers * B * n_heads * S * head_dim` elements: for this
model at full context (6 layers, 6 heads, hd 64, S=256, fp32, B=1) that is
2*6*1*6*256*64*4 bytes = 4.7 MB. Linear in S, and linear in batch -- which is
why long-context serving is a memory problem before it is a compute problem.

    python src/kv_cache.py --prompt "i think" --max-new-tokens 200
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from generate_naive import load_checkpoint, sample_from_logits
from model import GPT
from tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent.parent

KVCache = list[tuple[torch.Tensor, torch.Tensor]]


# ---------------------------------------------------------------------------
# cache helpers
# ---------------------------------------------------------------------------

def cache_length(past_kvs: KVCache | None) -> int:
    """Number of tokens currently in the cache (0 if empty)."""
    if not past_kvs:
        return 0
    return past_kvs[0][0].size(2)


def trim_cache(past_kvs: KVCache, length: int) -> KVCache:
    """Truncate every layer's K/V to the first `length` positions.

    Unused in Phase 2 -- generation only ever appends. It exists for Phase 3,
    where the model runs over speculative draft tokens that may be rejected, and
    their K/V entries have to be dropped before the next step. Leaving them in is
    the cache-contamination bug: the cache would then describe a sequence that
    differs from the one actually emitted, and every subsequent token would be
    conditioned on tokens that were never generated.
    """
    assert length <= cache_length(past_kvs), (
        f"cannot trim to {length}, cache only holds {cache_length(past_kvs)}"
    )
    return [(k[:, :, :length, :], v[:, :, :length, :]) for k, v in past_kvs]


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_cached(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    greedy: bool = False,
    return_logits: bool = False,
):
    """idx: (B, T) prompt token ids. Returns (B, T + max_new_tokens).

    Two phases, and they have genuinely different performance characteristics:

      PREFILL  - one forward pass over the whole prompt. Compute-bound: it is a
                 real (T x T) attention, same work the naive path does once.
      DECODE   - one forward pass per new token, feeding a single token and
                 attending it against the cache. Memory-bound: barely any FLOPs,
                 dominated by reading weights and the growing cache out of HBM.

    This split is why "tokens/sec" for a generator is not one number -- it
    depends on the prompt/generation ratio. The benchmark holds both fixed.
    """
    block_size = model.config.block_size
    prompt_len = idx.size(1)

    # Hard ceiling, and an honest one. The naive path handles running past
    # block_size by cropping context and recomputing every position; a cache
    # cannot, because cropping renumbers all positions and invalidates every
    # cached K/V at once. Sliding-window caches exist and re-encode positions
    # relatively (RoPE makes this natural), but with learned absolute embeddings
    # the only correct options are recompute or stop.
    if prompt_len + max_new_tokens > block_size:
        raise ValueError(
            f"prompt ({prompt_len}) + max_new_tokens ({max_new_tokens}) = "
            f"{prompt_len + max_new_tokens} exceeds block_size ({block_size}). "
            f"Cached generation cannot slide the window with learned absolute "
            f"positional embeddings -- shorten the prompt or generate fewer tokens."
        )

    step_logits = []

    # ---- prefill ----------------------------------------------------------
    logits, _, past_kvs = model(idx, use_cache=True)
    last = logits[:, -1, :]                                     # (B, vocab)

    for _ in range(max_new_tokens):
        if return_logits:
            step_logits.append(last.clone())
        next_id = sample_from_logits(last, temperature, top_k, greedy)
        idx = torch.cat([idx, next_id], dim=1)

        # ---- decode: feed ONLY the new token -------------------------------
        # (B, 1) in, not (B, T). The cache supplies everything the new token
        # needs to attend to. This is the whole optimization.
        logits, _, past_kvs = model(next_id, past_kvs=past_kvs, use_cache=True)
        last = logits[:, -1, :]

        # The cache must describe exactly the sequence emitted so far. If these
        # ever diverge, generation silently conditions on the wrong history --
        # so assert it rather than trusting the loop.
        assert cache_length(past_kvs) == idx.size(1), (
            f"cache holds {cache_length(past_kvs)} tokens but sequence is "
            f"{idx.size(1)} long"
        )

    return (idx, step_logits) if return_logits else idx


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "best.pt"))
    p.add_argument("--tokenizer", default=str(ROOT / "data" / "tokenizer.json"))
    p.add_argument("--prompt", default="\n")
    p.add_argument("--max-new-tokens", type=int, default=200)
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
    out = generate_cached(
        model, idx, args.max_new_tokens,
        temperature=args.temperature, top_k=args.top_k, greedy=args.greedy,
    )
    if args.device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    final_len = out.size(1)
    kv_mb = (
        2 * cfg.n_layers * 1 * cfg.n_heads * final_len * cfg.head_dim * 4 / 1e6
    )

    print("-" * 70)
    print(tok.decode(out[0].tolist()))
    print("-" * 70)
    print(
        f"cached: {args.max_new_tokens} tokens in {dt:.2f}s "
        f"= {args.max_new_tokens / dt:.1f} tok/s"
    )
    print(f"final cache: {final_len} tokens x {cfg.n_layers} layers = {kv_mb:.1f} MB (fp32)")


if __name__ == "__main__":
    main()
