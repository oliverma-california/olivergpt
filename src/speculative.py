"""
Speculative decoding with a training-free n-gram (prompt-lookup) drafter.

The idea: most of what a small model generates has appeared before in its own
context. So instead of a second neural "draft model", look at the last n tokens,
find where that n-gram last occurred in the sequence, and propose whatever
followed it. Costs nothing to train and microseconds to run.

The win is NOT fewer FLOPs -- verifying k draft tokens costs strictly more
arithmetic than decoding one token. The win is fewer *forward passes*: one
forward can confirm up to k+1 tokens. That matters precisely because single-
stream decoding on a small model is launch-overhead bound (see the Phase 2
numbers: 2.29ms at T=1 vs 2.75ms at T=256 -- the marginal token is nearly free).

    python src/speculative.py --prompt "i think" --max-new-tokens 200 --debug
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from generate_naive import load_checkpoint
from kv_cache import cache_length, trim_cache
from model import GPT
from tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 3a. the draft mechanism
# ---------------------------------------------------------------------------

def propose_draft(seq: list[int], n: int, k: int) -> list[int]:
    """Prompt-lookup draft: what followed the last occurrence of the last n tokens?

    seq: the full token sequence so far (prompt + generated).
    n:   lookup window -- how many trailing tokens must match.
    k:   maximum number of draft tokens to propose.

    Returns up to k tokens, or [] if the n-gram never occurred before (in which
    case the caller falls back to ordinary single-token decoding for that step).

    Searches BACKWARD, i.e. returns the continuation of the most *recent* match
    rather than the first. Recency is a better predictor: in a Discord corpus the
    thing you just said two lines ago is a likelier continuation than something
    from the top of the context.
    """
    if len(seq) < n + 1 or k <= 0:
        return []

    arr = np.asarray(seq)
    target = arr[-n:]

    # All length-n windows, as a (L-n+1, n) view -- no copying.
    windows = np.lib.stride_tricks.sliding_window_view(arr, n)
    matches = (windows == target).all(axis=1)

    # Drop the final window: that IS the suffix we are searching for, and
    # "matching itself" would propose the tokens that follow it -- which do not
    # exist yet. Everything strictly before it is a genuine earlier occurrence.
    matches = matches[:-1]

    hits = np.flatnonzero(matches)
    if hits.size == 0:
        return []

    i = int(hits[-1])                       # most recent occurrence
    return seq[i + n : i + n + k]


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

@dataclass
class SpecStats:
    """Per-run counters. The acceptance rate is the headline number: it bounds
    the achievable speedup, since tokens/forward can never exceed k+1."""
    steps: int = 0                       # forward passes through the target model
    tokens: int = 0                      # tokens actually emitted
    drafted: int = 0                     # draft tokens proposed, total
    accepted: int = 0                    # draft tokens accepted, total
    steps_with_draft: int = 0            # steps where the n-gram lookup hit
    accepted_per_step: list[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        """Fraction of *proposed* draft tokens that survived verification."""
        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def draft_hit_rate(self) -> float:
        """Fraction of steps where the lookup found anything to propose at all."""
        return self.steps_with_draft / self.steps if self.steps else 0.0

    @property
    def tokens_per_forward(self) -> float:
        """The quantity that actually converts into wall-clock speedup."""
        return self.tokens / self.steps if self.steps else 0.0

    def summary(self) -> str:
        return (
            f"{self.tokens} tokens in {self.steps} forward passes "
            f"({self.tokens_per_forward:.2f} tok/forward)\n"
            f"  draft hit rate:  {self.draft_hit_rate:6.1%} "
            f"({self.steps_with_draft}/{self.steps} steps found an n-gram match)\n"
            f"  acceptance rate: {self.acceptance_rate:6.1%} "
            f"({self.accepted}/{self.drafted} proposed tokens accepted)"
        )


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------

def _filtered_probs(logits: torch.Tensor, temperature: float, top_k: int | None):
    """(..., V) logits -> (..., V) probabilities, after temperature and top-k.

    Kept separate from sampling because speculative decoding needs the
    *distribution* itself, not just a draw from it: the acceptance test compares
    p(draft_token) against a uniform, and rejection needs the residual. Batched
    over leading dims so a whole verification step's rows can be filtered at once.
    """
    logits = logits / max(temperature, 1e-8)
    if top_k is not None:
        kk = min(top_k, logits.size(-1))
        kth = torch.topk(logits, kk, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    return F.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# 3b. the verify-and-accept loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_speculative(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    n: int = 4,
    k: int = 8,
    temperature: float = 1.0,
    top_k: int | None = None,
    greedy: bool = False,
    debug: bool = False,
    tokenizer: CharTokenizer | None = None,
):
    """idx: (1, T) prompt. Returns (out_idx, SpecStats).

    Batch size must be 1. Each sequence in a batch would accept a different
    number of draft tokens per step, so the cache would need per-sequence
    lengths -- that is ragged-cache bookkeeping (what paged-attention exists to
    solve) and well beyond the point of this project.

    THE LOOP INVARIANT, which is the whole ballgame:

        cache holds K/V for seq[0 : len(seq)-1]

    i.e. exactly one token is always "pending" -- the most recently emitted one,
    which has not been through the model yet. Every branch below restores this,
    and it is asserted every step. If the cache ever holds K/V for a token that
    is not in seq (a rejected draft token), every subsequent token is silently
    conditioned on a history that was never generated. That is the cache
    contamination bug, and it does not announce itself: output stays fluent, it
    is just wrong.
    """
    assert idx.size(0) == 1, "speculative decoding here is batch-1 only"
    block_size = model.config.block_size
    device = idx.device
    stats = SpecStats()

    # The token sequence is mirrored as a plain Python list. The drafter needs to
    # scan it every step, and doing that on the GPU tensor would force a
    # device->host sync per step -- which on this model costs about as much as
    # the forward pass we are trying to save.
    seq: list[int] = idx[0].tolist()

    if len(seq) + max_new_tokens > block_size:
        raise ValueError(
            f"prompt ({len(seq)}) + max_new_tokens ({max_new_tokens}) exceeds "
            f"block_size ({block_size}); same ceiling as kv_cache.generate_cached"
        )

    def emit(logits_row: torch.Tensor) -> int:
        """Pick a token from a (V,) logits row, honouring greedy/temp/top_k."""
        if greedy:
            return int(logits_row.argmax())
        probs = _filtered_probs(logits_row, temperature, top_k)
        return int(torch.multinomial(probs, 1))

    # ---- prefill: process the prompt, emit the first token -----------------
    logits, _, cache = model(idx, use_cache=True)
    first = emit(logits[0, -1])
    seq.append(first)
    stats.steps += 1
    stats.tokens += 1
    # cache_len == len(prompt) == len(seq) - 1. Invariant holds.

    while stats.tokens < max_new_tokens:
        C = cache_length(cache)
        assert C == len(seq) - 1, f"invariant broken: cache {C}, seq {len(seq)}"

        # ---- propose ------------------------------------------------------
        # Room check: we feed 1 pending token + j drafts, and C+1+j must stay
        # within block_size. Also never draft more tokens than we still need.
        budget = min(k, block_size - C - 1, max_new_tokens - stats.tokens)
        draft = propose_draft(seq, n, budget)
        j = len(draft)

        # ---- verify: ONE forward pass over [pending] + draft ---------------
        # Fed at absolute positions C .. C+j. This is the parallelism win:
        # j+1 positions scored for the price of one forward pass.
        inp = torch.tensor([[seq[C]] + draft], dtype=torch.long, device=device)
        logits, _, cache = model(inp, past_kvs=cache, use_cache=True)
        # logits[0, i] is the model's distribution for absolute position C+i+1.
        preds = logits[0]                                    # (1+j, V)

        # ---- accept -------------------------------------------------------
        # Accept the longest prefix of the draft the target model agrees with.
        # Position i is only meaningful if draft[0:i] were all accepted -- the
        # logits at C+i+1 were computed conditioned on the draft tokens before
        # it, so a mismatch invalidates everything after it. Hence: stop at the
        # first rejection, no "skip ahead and re-check" is possible.
        a = 0
        if greedy:
            argmaxes = preds.argmax(dim=-1)                  # (1+j,)
            while a < j and int(argmaxes[a]) == draft[a]:
                a += 1
            corrected = int(argmaxes[a])
        else:
            # Distribution-preserving acceptance. The drafter is deterministic,
            # so q(draft) = 1 and the standard min(1, p/q) test collapses to
            # "accept with probability p(draft_token)". On rejection, resample
            # from the normalized residual max(0, p - q), which is p with the
            # rejected token zeroed out. Together these leave the output
            # distribution exactly equal to ordinary sampling from p -- accepting
            # whenever the draft merely *could* have been sampled would bias
            # generation toward whatever the n-gram lookup happens to propose.
            probs_all = _filtered_probs(preds, temperature, top_k)     # (1+j, V)
            if j:
                # Draw every acceptance decision up front in one shot. Done one
                # token at a time this needs a device->host sync per draft token,
                # which on this model costs about as much as the forward pass
                # being saved. Uniforms are independent, so pre-drawing them is
                # exactly equivalent to drawing them lazily.
                p_draft = probs_all[torch.arange(j, device=device),
                                    torch.tensor(draft, device=device)]
                accept = (torch.rand(j, device=device) < p_draft).tolist()  # 1 sync
                while a < j and accept[a]:
                    a += 1
            if a < j:
                residual = probs_all[a].clone()
                residual[draft[a]] = 0.0
                total = float(residual.sum())
                # If p was a point mass on the rejected token, the residual is
                # empty; falling back to p is the only sensible choice and it
                # cannot loop, since that token would have been accepted w.p. 1.
                corrected = (
                    int(torch.multinomial(residual / total, 1)) if total > 0
                    else int(torch.multinomial(probs_all[a], 1))
                )
            else:
                corrected = int(torch.multinomial(probs_all[a], 1))   # bonus token

        # a draft tokens accepted, plus one token that is always correct:
        # either the model's replacement for the first rejected draft token, or
        # (if the whole draft was accepted) a free bonus token from the last
        # position, which we already paid to compute.
        new_tokens = draft[:a] + [corrected]

        if debug:
            _debug_step(stats.steps, seq, draft, a, corrected, tokenizer)

        seq.extend(new_tokens)

        # ---- repair the cache ---------------------------------------------
        # The forward pass wrote K/V for all 1+j fed tokens, at positions
        # C .. C+j. Only C .. C+a correspond to tokens we actually kept
        # (the pending token, plus the a accepted drafts). Positions C+a+1
        # onward are rejected drafts and MUST go: their K/V would otherwise be
        # attended to by every future token as if they had been generated.
        # The corrected token is deliberately NOT in the cache -- it becomes the
        # next step's pending token, restoring the invariant.
        cache = trim_cache(cache, C + 1 + a)

        stats.steps += 1
        stats.tokens += len(new_tokens)
        stats.drafted += j
        stats.accepted += a
        stats.steps_with_draft += 1 if j > 0 else 0
        stats.accepted_per_step.append(a)

        assert cache_length(cache) == len(seq) - 1, (
            f"cache contamination: cache {cache_length(cache)}, seq {len(seq)}"
        )

    # We may overshoot by up to k tokens on the final step; trim to the request.
    out = torch.tensor([seq], dtype=torch.long, device=device)
    stats.tokens = min(stats.tokens, max_new_tokens)
    return out[:, : idx.size(1) + max_new_tokens], stats


# ---------------------------------------------------------------------------
# 3c. debugging aid
# ---------------------------------------------------------------------------

def _debug_step(step, seq, draft, a, corrected, tok):
    """Print one step's draft / accept / reject decision, readably."""
    def show(ids):
        if tok is None:
            return str(ids)
        return repr(tok.decode(ids))[1:-1] if ids else ""

    ctx = show(seq[-12:])
    if not draft:
        print(f"  step {step:3d}  ctx ...{ctx!r:>16}  no n-gram match      -> {show([corrected])!r}")
        return
    verdict = "ALL ACCEPTED +bonus" if a == len(draft) else f"accepted {a}/{len(draft)}"
    print(
        f"  step {step:3d}  ctx ...{ctx!r:>16}  draft {show(draft)!r:<14} "
        f"{verdict:<19} -> {show(draft[:a] + [corrected])!r}"
    )


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "best.pt"))
    p.add_argument("--tokenizer", default=str(ROOT / "data" / "tokenizer.json"))
    p.add_argument("--prompt", default="i think")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("-n", "--ngram", type=int, default=4, help="lookup window size")
    p.add_argument("-k", "--draft-len", type=int, default=8, help="max draft tokens")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--debug", action="store_true", help="print every accept/reject")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    model, cfg, ckpt = load_checkpoint(args.ckpt, args.device)
    tok = CharTokenizer.load(args.tokenizer)
    idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=args.device)

    print(
        f"loaded {args.ckpt} (step {ckpt['step']}, val {ckpt['val_loss']:.4f}) | "
        f"n={args.ngram} k={args.draft_len} | "
        f"{'greedy' if args.greedy else f'temp {args.temperature} top-k {args.top_k}'}"
    )

    if args.device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out, stats = generate_speculative(
        model, idx, args.max_new_tokens,
        n=args.ngram, k=args.draft_len,
        temperature=args.temperature, top_k=args.top_k, greedy=args.greedy,
        debug=args.debug, tokenizer=tok,
    )
    if args.device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    print("-" * 70)
    print(tok.decode(out[0].tolist()))
    print("-" * 70)
    print(stats.summary())
    print(f"  wall: {dt:.3f}s = {stats.tokens / dt:.1f} tok/s")


if __name__ == "__main__":
    main()
