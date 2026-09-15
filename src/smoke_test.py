"""
Pre-training sanity checks for model.py. Run this before burning GPU time.

Checks, in order of how much they would hurt to get wrong:
  1. forward pass returns (batch, seq_len, vocab_size)
  2. loss at init is ~ln(vocab_size) -- i.e. the model starts out uniform
  3. causality: changing token t cannot change the logits at positions < t
  4. the model can overfit a single tiny batch in ~200 steps
"""

from __future__ import annotations

import math

import torch

from model import GPT, GPTConfig

torch.manual_seed(1337)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tiny config -- this test is about correctness, not capacity.
cfg = GPTConfig(vocab_size=65, d_model=64, n_heads=4, n_layers=2, block_size=32, dropout=0.0)
B, T = 4, 16


def check(name: str, ok: bool) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        raise SystemExit(1)


def main() -> None:
    print(f"device: {DEVICE}")
    model = GPT(cfg).to(DEVICE)
    print(f"params: {model.num_params():,}")

    idx = torch.randint(0, cfg.vocab_size, (B, T), device=DEVICE)
    targets = torch.randint(0, cfg.vocab_size, (B, T), device=DEVICE)

    # --- 1. output shape --------------------------------------------------
    model.eval()
    with torch.no_grad():
        logits, loss, _ = model(idx, targets)
    check(
        f"output shape {tuple(logits.shape)} == (B, T, vocab_size)",
        logits.shape == (B, T, cfg.vocab_size),
    )

    # --- 2. init loss ~ ln(vocab_size) ------------------------------------
    # A correctly initialized LM predicts roughly uniform over the vocab, so
    # cross-entropy should start near ln(V). Far above means the init is broken;
    # far below at step 0 means something is leaking the targets.
    expected = math.log(cfg.vocab_size)
    check(
        f"init loss {loss.item():.3f} ~= ln(vocab)={expected:.3f}",
        abs(loss.item() - expected) < 0.5,
    )

    # --- 3. causality -----------------------------------------------------
    # The single most important property to test, and the easiest to get wrong
    # (an off-by-one in the mask still trains, just with a leak). Perturb the
    # LAST token and confirm nothing before it moves.
    with torch.no_grad():
        base, _, _ = model(idx)
        perturbed_idx = idx.clone()
        perturbed_idx[:, -1] = (perturbed_idx[:, -1] + 1) % cfg.vocab_size
        perturbed, _, _ = model(perturbed_idx)
    prefix_same = torch.allclose(base[:, :-1], perturbed[:, :-1], atol=1e-6)
    last_changed = not torch.allclose(base[:, -1], perturbed[:, -1], atol=1e-6)
    check("causal mask: earlier positions unaffected by a later token", prefix_same)
    check("causal mask: the changed position itself does move", last_changed)

    # --- 4. can it learn? -------------------------------------------------
    # Overfitting one fixed batch is the cheapest end-to-end test that the
    # backward pass, optimizer wiring and loss are all connected.
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for step in range(200):
        _, l, _ = model(idx, targets)
        opt.zero_grad(set_to_none=True)
        l.backward()
        opt.step()
        losses.append(l.item())
        if step % 50 == 0:
            print(f"    step {step:3d}  loss {l.item():.4f}")
    print(f"    step 199  loss {losses[-1]:.4f}")
    check(
        f"loss decreased {losses[0]:.3f} -> {losses[-1]:.3f} on a fixed batch",
        losses[-1] < losses[0] * 0.2,
    )

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
