"""
Prove the KV cache changes nothing but the speed.

The cache is only a legitimate optimization if it is mathematically a no-op:
same weights, same arithmetic, just not repeated. So this asserts equality five
ways, from the weakest claim to the strongest:

  1. TOKENS       greedy generation produces identical token sequences
  2. LOGITS       per-step logits match within tolerance (much stricter than 1 --
                  argmax can agree while the underlying logits differ a lot)
  3. CHUNKING     feeding a prompt in chunks through the cache equals feeding it
                  whole (exercises the past_len > 0, T > 1 path Phase 3 needs)
  4. TRIM         a cache truncated to length M equals one built from scratch on
                  the first M tokens (the Phase 3 reject path)
  5. SPECULATIVE  greedy speculative decoding produces identical tokens to greedy
                  cached decoding, across a sweep of (n, k) draft settings

Tests 3 and 4 are not needed for Phase 2 -- decode-with-cache only ever feeds one
token and only ever appends. They are here because Phase 3 depends on both, and
finding out the rectangular-mask offset is wrong while also debugging a
verify-and-accept loop is a much worse afternoon than finding out now.

    python src/verify_correctness.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from generate_naive import generate_naive, load_checkpoint
from kv_cache import cache_length, generate_cached, trim_cache
from speculative import generate_speculative
from tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent.parent

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# ---------------------------------------------------------------------------

def test_tokens_and_logits(model, tok, prompt, n_tokens, device, tol):
    """Greedy decode both ways from the same prompt and compare."""
    print(f"\n[1/5 + 2/5] naive vs cached greedy decoding, {n_tokens} tokens")

    ids = tok.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    # Greedy (argmax) on both sides, so there is no RNG to keep in sync and any
    # difference that shows up is a real numerical difference, not a different
    # draw. Seeds are set anyway so the run is reproducible end to end.
    torch.manual_seed(1337)
    naive_out, naive_logits = generate_naive(
        model, idx, n_tokens, greedy=True, return_logits=True
    )
    torch.manual_seed(1337)
    cached_out, cached_logits = generate_cached(
        model, idx, n_tokens, greedy=True, return_logits=True
    )

    same_tokens = torch.equal(naive_out, cached_out)
    record(
        "token sequences identical",
        same_tokens,
        f"{naive_out.size(1)} tokens" if same_tokens
        else f"first divergence at position {int((naive_out != cached_out).float().argmax())}",
    )

    # Stack the per-step last-position logits: (n_tokens, B, vocab).
    a = torch.stack(naive_logits)
    b = torch.stack(cached_logits)
    max_abs = (a - b).abs().max().item()
    record(
        f"per-step logits match (atol={tol})",
        torch.allclose(a, b, atol=tol, rtol=0),
        f"max abs diff {max_abs:.3e}",
    )

    if same_tokens:
        print("\n  generated text (identical from both paths):")
        text = tok.decode(cached_out[0].tolist())
        for line in text.splitlines():
            print(f"    | {line}")

    return max_abs


def test_chunked_prefill(model, tok, prompt, device, tol):
    """Feeding a prompt in two chunks must equal feeding it in one.

    This is the multi-token-with-nonempty-cache path: chunk 2 arrives with
    past_len = len(chunk 1) and T = len(chunk 2) > 1, so the causal mask really
    is a rectangle sliced out of the middle of the triangle. Get the row offset
    wrong and this fails while single-token decoding still passes -- because at
    T=1 the mask row is all-True and masking is a no-op.
    """
    print("\n[3/5] chunked prefill vs single-shot prefill")

    ids = tok.encode(prompt)
    assert len(ids) >= 4, "need a prompt of at least 4 characters for this test"
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    split = len(ids) // 2

    with torch.no_grad():
        full_logits, _, _ = model(idx, use_cache=True)

        first, _, kvs = model(idx[:, :split], use_cache=True)
        second, _, kvs = model(idx[:, split:], past_kvs=kvs, use_cache=True)
        chunked_logits = torch.cat([first, second], dim=1)

    max_abs = (full_logits - chunked_logits).abs().max().item()
    record(
        f"chunked prefill matches (atol={tol})",
        torch.allclose(full_logits, chunked_logits, atol=tol, rtol=0),
        f"max abs diff {max_abs:.3e}, split {split}/{len(ids)}",
    )
    record(
        "cache length after chunked prefill == prompt length",
        cache_length(kvs) == len(ids),
        f"{cache_length(kvs)} == {len(ids)}",
    )


def test_cache_trim(model, tok, prompt, device, tol):
    """A trimmed cache must equal one built from scratch on the shorter prefix.

    Phase 3 will speculatively push draft tokens through the model, then throw
    the rejected ones away by trimming. That is only sound if a trimmed cache is
    indistinguishable from one that never saw the rejected tokens -- which it
    should be, because attention is causal and nothing later can alter an
    earlier position's K/V. This asserts that rather than assuming it.
    """
    print("\n[4/5] trimmed cache vs freshly built cache")

    ids = tok.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    keep = len(ids) // 2

    with torch.no_grad():
        _, _, full_kvs = model(idx, use_cache=True)
        trimmed = trim_cache(full_kvs, keep)
        _, _, fresh_kvs = model(idx[:, :keep], use_cache=True)

    worst = max(
        max((tk - fk).abs().max().item(), (tv - fv).abs().max().item())
        for (tk, tv), (fk, fv) in zip(trimmed, fresh_kvs)
    )
    record(
        f"trimmed K/V matches fresh K/V across all {len(trimmed)} layers (atol={tol})",
        all(
            torch.allclose(tk, fk, atol=tol, rtol=0) and torch.allclose(tv, fv, atol=tol, rtol=0)
            for (tk, tv), (fk, fv) in zip(trimmed, fresh_kvs)
        ),
        f"max abs diff {worst:.3e}",
    )

    # And the trimmed cache must actually continue correctly, not just look right.
    with torch.no_grad():
        from_trimmed, _, _ = model(idx[:, keep:], past_kvs=trimmed, use_cache=True)
        from_fresh, _, _ = model(idx[:, keep:], past_kvs=fresh_kvs, use_cache=True)
    record(
        "continuing from a trimmed cache gives the same logits",
        torch.allclose(from_trimmed, from_fresh, atol=tol, rtol=0),
        f"max abs diff {(from_trimmed - from_fresh).abs().max().item():.3e}",
    )


# ---------------------------------------------------------------------------

def test_speculative(model, tok, prompt, n_tokens, device):
    """Greedy speculative decoding must be token-identical to greedy cached.

    This is the strongest claim in the project and the easiest to get subtly
    wrong. Speculative decoding is only a legitimate optimization if it is
    *exactly* the same computation reordered -- if it changes even one token, it
    is not an optimization, it is a different (worse) sampler wearing a costume.

    Swept over n and k rather than tested at one setting, because the failure
    modes are shape-dependent: k=1 never exercises multi-token accept, large k
    stresses the block_size clamp, and small n produces the frequent-but-wrong
    drafts that expose an off-by-one in the accept count or the cache trim.
    """
    print(f"\n[5/5] greedy speculative vs greedy cached, {n_tokens} tokens")

    ids = tok.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    reference = generate_cached(model, idx, n_tokens, greedy=True)

    mismatches = []
    for n in (2, 3, 4, 6, 8):
        for k in (1, 2, 4, 8, 16):
            out, _ = generate_speculative(model, idx, n_tokens, n=n, k=k, greedy=True)
            if not torch.equal(reference, out):
                pos = int((reference != out).float().argmax())
                mismatches.append(f"n={n},k={k}@{pos}")

    record(
        "speculative == cached across 25 (n, k) settings",
        not mismatches,
        "all identical" if not mismatches else f"differs at {', '.join(mismatches[:4])}",
    )

    # A draft that is never accepted must still be correct -- it just degenerates
    # to one token per forward pass. k=0 is the degenerate case that proves the
    # fallback path (no draft proposed) does not disturb the invariant.
    out, stats = generate_speculative(model, idx, n_tokens, n=4, k=0, greedy=True)
    record(
        "k=0 (no drafting) degenerates to plain cached decoding",
        torch.equal(reference, out) and stats.tokens_per_forward == 1.0,
        f"{stats.tokens_per_forward:.2f} tok/forward",
    )

    _, stats = generate_speculative(model, idx, n_tokens, n=4, k=8, greedy=True)
    print(f"\n  at n=4, k=8: {stats.summary()}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "best.pt"))
    p.add_argument("--tokenizer", default=str(ROOT / "data" / "tokenizer.json"))
    p.add_argument("--prompt", default="i think that")
    p.add_argument("--n-tokens", type=int, default=50)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    # Force true fp32 matmuls. TF32 (which train.py deliberately turns on) has a
    # 10-bit mantissa, and the two code paths reduce over different shapes, so
    # under TF32 the differences here would be ~1e-2 and this test would be
    # measuring the numerics of the GPU rather than the correctness of the cache.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model, cfg, ckpt = load_checkpoint(args.ckpt, args.device)
    tok = CharTokenizer.load(args.tokenizer)
    print(
        f"checkpoint: {args.ckpt} (step {ckpt['step']}, val {ckpt['val_loss']:.4f})\n"
        f"device: {args.device} | fp32 (TF32 disabled) | prompt: {args.prompt!r}"
    )

    test_tokens_and_logits(model, tok, args.prompt, args.n_tokens, args.device, args.tol)
    test_chunked_prefill(model, tok, args.prompt, args.device, args.tol)
    test_cache_trim(model, tok, args.prompt, args.device, args.tol)
    test_speculative(model, tok, args.prompt, args.n_tokens, args.device)

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print("\n" + "=" * 70)
    if passed == total:
        print(f"PASS -- {passed}/{total} checks. Caching and speculation are "
              f"no-ops on the\n        output: same tokens, same logits.")
    else:
        print(f"FAIL -- {passed}/{total} checks passed:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"    - {name}  {detail}")
    print("=" * 70)
    raise SystemExit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
