"""
Phase 4: the unified benchmark. Naive vs KV-cached vs speculative decoding.

Produces results/benchmark_results.json and results/benchmark_plot.png.

Deliberately measures TWO batch sizes, because the headline "which optimization
wins" answer inverts between them and reporting only one would be misleading:

  batch 1  -- launch-overhead bound. The KV cache does nothing (1.0x); only
              speculative decoding, which cuts the NUMBER of forward passes,
              helps.
  batch 64 -- compute bound. The KV cache is worth up to 7x; speculative
              decoding does not apply (batch-1 only, see speculative.py).

And two decoding modes, because speculative decoding's speedup is a property of
the decoding mode as much as of the implementation: greedy is deterministic so
drafts are often right, while sampling injects entropy at every position.

    python src/benchmark.py
    python src/benchmark.py --skip-heavy      # omit the batch-64 sweep
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from generate_naive import generate_naive, load_checkpoint
from kv_cache import generate_cached
from speculative import generate_speculative
from tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent.parent

# Reference categorical palette, slots 1-3. This subset is the one documented as
# clearing the all-pairs colorblind gates in both light and dark modes; every bar
# also carries a direct value label, so identity never rests on color alone.
C_NAIVE, C_CACHED, C_SPEC = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#dcdbd6"
SURFACE = "#fcfcfb"


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------

def timed(fn, model, idx, n_tokens, reps, device, **kw):
    """Warm up once, then time `reps` runs. Returns (mean_s, best_s, last_result).

    The warmup is not optional: the first call pays CUDA context setup, kernel
    autotuning and allocator growth, which on a run this short would dominate.
    Mean is reported (the plan asks for an average); best is recorded too, since
    for a deterministic workload the spread is measurement noise and the minimum
    is the better estimate of true cost.
    """
    fn(model, idx, 16, **kw)
    if device == "cuda":
        torch.cuda.synchronize()
    times, result = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        result = fn(model, idx, n_tokens, **kw)
        if device == "cuda":
            torch.cuda.synchronize()   # CUDA is async; stop the clock honestly
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times), min(times), result


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------

def make_plot(data, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.flat:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)   # recessive grid
        ax.set_axisbelow(True)

    # -- (a) batch 1, three methods x two decoding modes ---------------------
    ax = axes[0, 0]
    modes = [m["mode"] for m in data["batch1"]]
    methods = [("naive", C_NAIVE), ("cached", C_CACHED), ("speculative", C_SPEC)]
    x = range(len(modes))
    w = 0.26
    for i, (name, color) in enumerate(methods):
        vals = [m[name]["tok_per_s"] for m in data["batch1"]]
        # 2px-equivalent gap between adjacent bars: width slightly under the slot
        pos = [xi + (i - 1) * w for xi in x]
        ax.bar(pos, vals, w * 0.92, label=name, color=color, edgecolor=SURFACE, linewidth=1.5)
        for p, v in zip(pos, vals):
            ax.text(p, v + 12, f"{v:.0f}", ha="center", va="bottom",
                    fontsize=8.5, color=INK_2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(modes, color=INK_2)
    ax.set_ylabel("tokens / sec", color=INK_2, fontsize=9)
    ax.set_title("Batch 1: only speculation helps\n"
                 "launch-overhead bound, so cutting FLOPs does nothing",
                 color=INK, fontsize=11, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, ncol=3,
              loc="upper left", bbox_to_anchor=(0, -0.10))
    ax.set_ylim(0, max(m["speculative"]["tok_per_s"] for m in data["batch1"]) * 1.22)

    # -- (b) batch 64, naive vs cached across generation length --------------
    ax = axes[0, 1]
    if data.get("batch64"):
        lens = [r["n_tokens"] for r in data["batch64"]]
        ax.plot(lens, [r["naive"]["tok_per_s"] for r in data["batch64"]],
                marker="o", markersize=8, linewidth=2, color=C_NAIVE, label="naive")
        ax.plot(lens, [r["cached"]["tok_per_s"] for r in data["batch64"]],
                marker="o", markersize=8, linewidth=2, color=C_CACHED, label="cached")
        for r in data["batch64"]:
            ax.text(r["n_tokens"], r["cached"]["tok_per_s"] * 1.04,
                    f"{r['cached']['tok_per_s']/r['naive']['tok_per_s']:.1f}x",
                    ha="center", fontsize=8.5, color=INK_2)
        ax.set_xticks(lens)
        # Zero baseline: this panel is a magnitude comparison, and an axis that
        # starts partway up would overstate how far the naive line falls.
        ax.set_ylim(0, max(r["cached"]["tok_per_s"] for r in data["batch64"]) * 1.15)
        ax.set_xlabel("tokens generated", color=INK_2, fontsize=9)
        ax.set_ylabel("tokens / sec (all 64 sequences)", color=INK_2, fontsize=9)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, ncol=2,
                  loc="upper left", bbox_to_anchor=(0, -0.10))
    else:
        ax.text(0.5, 0.5, "batch-64 sweep skipped", ha="center", va="center",
                color=INK_2, transform=ax.transAxes)
    ax.set_title("Batch 64: the cache is worth up to 7x\n"
                 "compute bound, and naive cost grows with context",
                 color=INK, fontsize=11, loc="left", pad=12)

    # -- (c) speculative: speedup vs lookup window n -------------------------
    ax = axes[1, 0]
    sweep = data["spec_sweep"]
    ns = [r["n"] for r in sweep]
    ax.bar(range(len(ns)), [r["speedup_vs_cached"] for r in sweep], 0.62,
           color=C_SPEC, edgecolor=SURFACE, linewidth=1.5)
    for i, r in enumerate(sweep):
        ax.text(i, r["speedup_vs_cached"] + 0.02, f"{r['speedup_vs_cached']:.2f}x",
                ha="center", va="bottom", fontsize=8.5, color=INK_2)
        ax.text(i, 0.06, f"hit\n{r['draft_hit_rate']:.0%}", ha="center",
                fontsize=7.5, color=SURFACE, weight="bold")
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels([f"n={n}" for n in ns], color=INK_2)
    ax.set_ylabel("speedup vs cached", color=INK_2, fontsize=9)
    ax.axhline(1.0, color=INK_2, linewidth=1, linestyle="--", alpha=0.5)
    ax.set_title("Speedup tracks the draft HIT rate, not acceptance\n"
                 "small n proposes often and wrong - and that wins",
                 color=INK, fontsize=11, loc="left", pad=12)

    # -- (d) headline stat tiles --------------------------------------------
    ax = axes[1, 1]
    ax.axis("off")
    g = next(m for m in data["batch1"] if m["mode"] == "greedy")
    tiles = [
        ("1.00x", "KV cache, batch 1", "no win: not FLOP-bound"),
        (f"{data['batch64'][-1]['cached']['tok_per_s'] / data['batch64'][-1]['naive']['tok_per_s']:.2f}x"
         if data.get("batch64") else "n/a", "KV cache, batch 64", "200 tokens, compute-bound"),
        (f"{g['speculative']['tok_per_s'] / g['cached']['tok_per_s']:.2f}x",
         "Speculative, greedy", f"{g['speculative']['tokens_per_forward']:.2f} tokens/forward"),
        (f"{max(r['speedup_vs_cached'] for r in sweep):.2f}x", "Best n setting",
         f"n={max(sweep, key=lambda r: r['speedup_vs_cached'])['n']}, k={data['spec_k']}"),
    ]
    for i, (big, label, sub) in enumerate(tiles):
        cx, cy = 0.06 + (i % 2) * 0.50, 0.72 - (i // 2) * 0.42
        ax.text(cx, cy, big, fontsize=30, color=INK, weight="bold", transform=ax.transAxes)
        ax.text(cx, cy - 0.11, label, fontsize=10, color=INK, transform=ax.transAxes)
        ax.text(cx, cy - 0.19, sub, fontsize=8.5, color=INK_2, transform=ax.transAxes)
    ax.set_title("Headline", color=INK, fontsize=11, loc="left", pad=12)

    meta = data["meta"]
    fig.suptitle(
        f"Char-level GPT ({meta['params']/1e6:.1f}M params) inference optimization  -  "
        f"{meta['device']}  -  {meta['n_tokens']} tokens, mean of {meta['reps']} runs",
        color=INK, fontsize=12.5, x=0.02, ha="left", y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    print(f"wrote {out_png}")


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "best.pt"))
    p.add_argument("--tokenizer", default=str(ROOT / "data" / "tokenizer.json"))
    p.add_argument("--prompt", default="i think")
    p.add_argument("--n-tokens", type=int, default=200)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--spec-n", type=int, default=2)
    p.add_argument("--spec-k", type=int, default=12)
    p.add_argument("--batch64-lens", type=int, nargs="+", default=[50, 100, 200])
    p.add_argument("--cooldown", type=float, default=5.0)
    p.add_argument("--skip-heavy", action="store_true",
                   help="skip the batch-64 sweep (the sustained-load part)")
    p.add_argument("--out-json", default=str(ROOT / "results" / "benchmark_results.json"))
    p.add_argument("--out-png", default=str(ROOT / "results" / "benchmark_plot.png"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--plot-only", action="store_true",
                   help="re-render the figure from an existing results JSON, no GPU work")
    args = p.parse_args()

    if args.plot_only:
        make_plot(json.loads(Path(args.out_json).read_text(encoding="utf-8")), args.out_png)
        return

    torch.manual_seed(1337)
    model, cfg, ckpt = load_checkpoint(args.ckpt, args.device)
    tok = CharTokenizer.load(args.tokenizer)
    ids = tok.encode(args.prompt)
    idx1 = torch.tensor([ids], dtype=torch.long, device=args.device)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "meta": {
            "device": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
            "torch": torch.__version__,
            "params": model.num_params(),
            "config": dict(cfg.__dict__),
            "prompt": args.prompt,
            "n_tokens": args.n_tokens,
            "reps": args.reps,
            "val_loss": ckpt["val_loss"],
        },
        "spec_n": args.spec_n,
        "spec_k": args.spec_k,
        "batch1": [],
        "batch64": [],
        "spec_sweep": [],
    }
    # Written after every section, not just at the end: an interrupted run should
    # still leave usable data behind.
    def flush():
        out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")

    N, R, dev = args.n_tokens, args.reps, args.device
    print(f"{data['meta']['device']} | {N} tokens | mean of {R} runs | prompt {args.prompt!r}\n")

    # ---- batch 1, both decoding modes -------------------------------------
    print("=== batch 1 ===")
    for mode, kw in (("greedy", dict(greedy=True)),
                     ("temp 0.8, top-k 40", dict(temperature=0.8, top_k=40))):
        row = {"mode": mode}
        for name, fn, extra in (
            ("naive", generate_naive, {}),
            ("cached", generate_cached, {}),
            ("speculative", generate_speculative, dict(n=args.spec_n, k=args.spec_k)),
        ):
            mean, best, res = timed(fn, model, idx1, N, R, dev, **kw, **extra)
            row[name] = {"mean_s": mean, "best_s": best, "tok_per_s": N / mean}
            if name == "speculative":
                st = res[1]
                row[name].update({
                    "tokens_per_forward": st.tokens_per_forward,
                    "acceptance_rate": st.acceptance_rate,
                    "draft_hit_rate": st.draft_hit_rate,
                })
            print(f"  {mode:<20} {name:<12} {N/mean:7.1f} tok/s")
        row["cache_speedup"] = row["naive"]["mean_s"] / row["cached"]["mean_s"]
        row["spec_speedup"] = row["cached"]["mean_s"] / row["speculative"]["mean_s"]
        print(f"  {'':<20} -> cache {row['cache_speedup']:.2f}x, "
              f"speculative {row['spec_speedup']:.2f}x vs cached\n")
        data["batch1"].append(row)
        flush()

    # ---- speculative n sweep (greedy) -------------------------------------
    print("=== speculative n sweep (greedy, k=%d) ===" % args.spec_k)
    _, cached_best, _ = timed(generate_cached, model, idx1, N, R, dev, greedy=True)
    cached_mean = data["batch1"][0]["cached"]["mean_s"]
    for n in (2, 3, 4, 6, 8, 10):
        mean, best, res = timed(generate_speculative, model, idx1, N, R, dev,
                                greedy=True, n=n, k=args.spec_k)
        st = res[1]
        rec = {
            "n": n, "k": args.spec_k, "mean_s": mean, "tok_per_s": N / mean,
            "speedup_vs_cached": cached_mean / mean,
            "tokens_per_forward": st.tokens_per_forward,
            "acceptance_rate": st.acceptance_rate,
            "draft_hit_rate": st.draft_hit_rate,
        }
        data["spec_sweep"].append(rec)
        flush()
        print(f"  n={n:<3} {N/mean:7.1f} tok/s  {rec['speedup_vs_cached']:.2f}x  "
              f"hit {st.draft_hit_rate:5.1%}  accept {st.acceptance_rate:5.1%}")

    # ---- batch 64, naive vs cached ----------------------------------------
    if not args.skip_heavy:
        print("\n=== batch 64 (naive vs cached; speculative is batch-1 only) ===")
        idx64 = idx1.expand(64, -1).contiguous()
        for n_tok in args.batch64_lens:
            row = {"n_tokens": n_tok}
            for name, fn in (("naive", generate_naive), ("cached", generate_cached)):
                if args.cooldown:
                    time.sleep(args.cooldown)
                mean, best, _ = timed(fn, model, idx64, n_tok, R, dev, greedy=True)
                row[name] = {"mean_s": mean, "best_s": best, "tok_per_s": n_tok * 64 / mean}
            row["speedup"] = row["naive"]["mean_s"] / row["cached"]["mean_s"]
            data["batch64"].append(row)
            flush()
            print(f"  {n_tok:>4} tokens  naive {row['naive']['tok_per_s']:8.0f}  "
                  f"cached {row['cached']['tok_per_s']:8.0f}  {row['speedup']:.2f}x")

    flush()
    print(f"\nwrote {out_json}")
    make_plot(data, args.out_png)


if __name__ == "__main__":
    main()
