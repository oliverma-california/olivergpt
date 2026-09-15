"""
Statistical check that speculative SAMPLING preserves the output distribution.

Greedy speculative decoding can be verified by exact token equality (see
verify_correctness.py). Sampling cannot: accepting or rejecting a draft token
consumes randomness differently, so two correct implementations produce
different sequences from the same seed. The claim to test is therefore
distributional, not exact.

Why this needs testing at all: the intuitive acceptance rule -- "accept the
draft token if the target model could plausibly have sampled it" -- is WRONG.
It biases generation toward whatever the n-gram lookup proposes, which on a
repetitive corpus means biasing toward more repetition. The correct rule
(Leviathan et al. 2023) accepts with probability min(1, p/q) and on rejection
resamples from the normalized residual max(0, p - q). Because a prompt-lookup
drafter is deterministic, q is a point mass and this collapses to: accept with
probability p(draft token); on rejection, resample from p with that token zeroed.

Method: draw many continuations from a fixed prompt under both cached sampling
and speculative sampling, then compare per-position empirical distributions by
total variation distance. TV between two independent runs of the SAME method
gives the noise floor -- the comparison is only meaningful against that, since
with finite samples nothing matches exactly.

    python src/verify_sampling.py --samples 3000
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch

from generate_naive import load_checkpoint
from kv_cache import generate_cached
from speculative import generate_speculative
from tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent.parent


def tv_distance(a: Counter, b: Counter) -> float:
    """Total variation distance between two empirical distributions."""
    na, nb = sum(a.values()), sum(b.values())
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a[x] / na - b[x] / nb) for x in keys)


def collect(fn, model, idx, n_new, samples, **kw):
    """Run fn `samples` times; return per-position Counters of emitted tokens."""
    prompt_len = idx.size(1)
    cols = [Counter() for _ in range(n_new)]
    for _ in range(samples):
        out = fn(model, idx, n_new, **kw)
        if isinstance(out, tuple):
            out = out[0]
        toks = out[0, prompt_len:].tolist()
        for i, t in enumerate(toks[:n_new]):
            cols[i][t] += 1
    return cols


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "best.pt"))
    p.add_argument("--tokenizer", default=str(ROOT / "data" / "tokenizer.json"))
    # A deliberately repetitive prompt, so the n-gram lookup hits immediately and
    # the accept/reject machinery is actually exercised at every position. On a
    # non-repetitive prompt the drafter proposes nothing and this would silently
    # be testing plain cached sampling against itself.
    p.add_argument("--prompt", default="the same time the same time the same ")
    p.add_argument("--n-new", type=int, default=6)
    p.add_argument("--samples", type=int, default=3000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("-n", "--ngram", type=int, default=4)
    p.add_argument("-k", "--draft-len", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    model, _, ckpt = load_checkpoint(args.ckpt, args.device)
    tok = CharTokenizer.load(args.tokenizer)
    idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=args.device)

    kw = dict(temperature=args.temperature, top_k=args.top_k)
    print(
        f"prompt {args.prompt!r}\n{args.samples} samples x {args.n_new} tokens | "
        f"temp {args.temperature} top-k {args.top_k} | n={args.ngram} k={args.draft_len}\n"
    )

    torch.manual_seed(1)
    ref_a = collect(generate_cached, model, idx, args.n_new, args.samples, **kw)
    torch.manual_seed(2)
    ref_b = collect(generate_cached, model, idx, args.n_new, args.samples, **kw)
    torch.manual_seed(3)
    spec = collect(
        generate_speculative, model, idx, args.n_new, args.samples,
        n=args.ngram, k=args.draft_len, **kw
    )

    print(f"{'pos':>4} {'TV(cached,cached)':>18} {'TV(cached,spec)':>17} {'ratio':>7}  verdict")
    worst_ratio = 0.0
    for i in range(args.n_new):
        floor = tv_distance(ref_a[i], ref_b[i])
        got = tv_distance(ref_a[i], spec[i])
        # Compare against the noise floor, not against zero. A ratio near 1 means
        # speculative sampling differs from cached sampling by no more than two
        # runs of cached sampling differ from each other.
        ratio = got / floor if floor > 0 else float("inf")
        worst_ratio = max(worst_ratio, ratio)
        print(f"{i:>4} {floor:>18.4f} {got:>17.4f} {ratio:>7.2f}  "
              f"{'ok' if ratio < 2.0 else 'SUSPICIOUS'}")

    print()
    if worst_ratio < 2.0:
        print(f"PASS -- worst position is {worst_ratio:.2f}x the noise floor (<2.0).")
        print("Speculative sampling is distributionally indistinguishable from")
        print("ordinary sampling at this sample size.")
    else:
        print(f"FAIL -- worst position is {worst_ratio:.2f}x the noise floor.")
        print("The acceptance rule is biasing the output distribution.")
    raise SystemExit(0 if worst_ratio < 2.0 else 1)


if __name__ == "__main__":
    main()
