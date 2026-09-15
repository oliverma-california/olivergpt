"""
Phase 2 benchmark: naive vs KV-cached generation, batch 1 and batched.

The point of running both batch sizes is that they land in different regimes and
the cache only pays off in one of them:

  batch 1   - launch-overhead bound. A forward pass costs ~2.3ms at T=1 and
              ~2.7ms at T=256 on this model, so cutting FLOPs per forward buys
              almost nothing: both paths issue the same NUMBER of forwards.
  batch 64  - compute bound. T=256 costs ~51ms vs ~2.5ms at T=1, so the
              quadratic recompute the naive path does is real work, and removing
              it is a real win.

Results are appended to the JSON file after every single measurement rather than
at the end, so an interrupted run (this benchmark previously took a laptop GPU
down with it) still leaves usable data behind.

    python src/bench_kv.py --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from generate_naive import generate_naive, load_checkpoint
from kv_cache import generate_cached
from tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent.parent

METHODS = {"naive": generate_naive, "cached": generate_cached}


def time_generation(fn, model, idx, n_tokens, reps, device):
    """Return (best, mean) seconds over `reps` timed runs, after a warmup.

    Reports best-of as the headline. For a deterministic workload with no input
    variation, run-to-run spread is measurement noise (scheduler, clocks, other
    processes), and the minimum is the closest estimate of the true cost.
    """
    fn(model, idx, 16, greedy=True)  # warmup: CUDA context, kernel autotune, allocator
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(model, idx, n_tokens, greedy=True)
        if device == "cuda":
            torch.cuda.synchronize()  # CUDA launches are async; stop the clock honestly
        times.append(time.perf_counter() - t0)
    return min(times), sum(times) / len(times)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "best.pt"))
    p.add_argument("--tokenizer", default=str(ROOT / "data" / "tokenizer.json"))
    p.add_argument("--prompt", default="i think that")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-tokens", type=int, nargs="+", default=[50, 100, 200])
    p.add_argument("--reps", type=int, default=3)
    p.add_argument(
        "--cooldown", type=float, default=5.0,
        help="Seconds idle between measurements. Does not affect the timings "
             "themselves (each is warmed up separately); it keeps a laptop GPU "
             "from sitting at 100%% continuously for the whole sweep.",
    )
    p.add_argument("--out", default=str(ROOT / "results" / "kv_benchmark.json"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(1337)
    model, cfg, ckpt = load_checkpoint(args.ckpt, args.device)
    tok = CharTokenizer.load(args.tokenizer)

    ids = tok.encode(args.prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=args.device)
    idx = idx.expand(args.batch_size, -1).contiguous()

    meta = {
        "device": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
        "torch": torch.__version__,
        "batch_size": args.batch_size,
        "prompt_len": len(ids),
        "reps": args.reps,
        "params": model.num_params(),
        "config": {k: v for k, v in cfg.__dict__.items()},
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    def flush():
        out_path.write_text(
            json.dumps({"meta": meta, "records": records}, indent=2), encoding="utf-8"
        )

    print(f"{meta['device']} | batch {args.batch_size} | prompt {len(ids)} tok | reps {args.reps}")
    print(f"{'n_tokens':>9} {'method':>8} {'best s':>9} {'tok/s':>10} {'tok/s/seq':>10}")
    flush()

    for n in args.n_tokens:
        for name, fn in METHODS.items():
            if args.cooldown:
                time.sleep(args.cooldown)
            best, mean = time_generation(fn, model, idx, n, args.reps, args.device)
            # Two rates: total tokens across the batch, and per-sequence latency.
            # Batching inflates aggregate throughput even when each sequence gets
            # slower, so quoting only the aggregate would flatter the batched runs.
            rec = {
                "n_tokens": n,
                "method": name,
                "best_s": best,
                "mean_s": mean,
                "tok_per_s_total": n * args.batch_size / best,
                "tok_per_s_per_seq": n / best,
            }
            records.append(rec)
            flush()
            print(
                f"{n:>9} {name:>8} {best:>9.3f} {rec['tok_per_s_total']:>10.1f} "
                f"{rec['tok_per_s_per_seq']:>10.1f}",
                flush=True,
            )

    print("\nspeedups (cached vs naive, same n_tokens):")
    for n in args.n_tokens:
        got = {r["method"]: r["best_s"] for r in records if r["n_tokens"] == n}
        if len(got) == 2:
            print(f"  n={n:<4} {got['naive'] / got['cached']:.2f}x")

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
