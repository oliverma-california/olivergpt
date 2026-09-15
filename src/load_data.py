"""
Build the char-level tokenizer from data/corpus.txt and save train/val tensors.

Outputs (all under data/):
    tokenizer.json  -- the vocab, so generation reloads the exact id mapping
    train.pt        -- 1-D uint16 tensor of token ids
    val.pt          -- ditto

The split is contiguous (first 90% train, last 10% val), not shuffled. The
corpus is one Discord message per line in chronological order, so a contiguous
split means validation is a *later* stretch of messages. That's the honest
setup: a shuffled split would leak adjacent lines from the same conversation
into both halves and flatter the val loss.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default=str(ROOT / "data" / "corpus.txt"))
    p.add_argument("--out-dir", default=str(ROOT / "data"))
    p.add_argument(
        "--min-char-freq", type=int, default=5,
        help="Drop characters appearing fewer than this many times in the corpus "
             "(default: 5). On a Discord export this prunes one-off emoji/CJK that "
             "would otherwise be untrainable dead rows in the embedding table.",
    )
    p.add_argument("--val-frac", type=float, default=0.1)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    text = Path(args.corpus).read_text(encoding="utf-8")
    print(f"corpus: {len(text):,} characters, {text.count(chr(10)) + 1:,} lines")

    tok = CharTokenizer.from_text(text, min_freq=args.min_char_freq)
    tok.save(out_dir / "tokenizer.json")
    print(f"vocab_size: {tok.vocab_size}  (min_char_freq={args.min_char_freq})")

    # drop_unknown=True silently strips the pruned rare characters.
    ids = tok.encode(text)
    dropped = len(text) - len(ids)
    print(f"encoded {len(ids):,} tokens ({dropped:,} chars dropped as too rare)")

    # Stored as int64 even though a char vocab fits in a uint8. At ~600K tokens
    # that is 5MB on disk -- not worth the narrower dtype, which would force a
    # cast in the hot batching path and pulls in torch's patchy uint16 support.
    # For a corpus 100x bigger, revisit this.
    data = torch.tensor(ids, dtype=torch.long)

    n_val = int(len(data) * args.val_frac)
    train, val = data[: len(data) - n_val], data[len(data) - n_val :]

    torch.save(train, out_dir / "train.pt")
    torch.save(val, out_dir / "val.pt")
    print(f"train: {len(train):,} tokens -> {out_dir / 'train.pt'}")
    print(f"val:   {len(val):,} tokens -> {out_dir / 'val.pt'}")


if __name__ == "__main__":
    main()
