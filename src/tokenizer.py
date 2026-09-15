"""
Character-level tokenizer.

Deliberately minimal: the vocabulary is just the set of distinct characters in
the training corpus, so `encode`/`decode` are dictionary lookups and there is no
subword merging step at all. For a ~600KB personal corpus this is the right
call -- a BPE vocab would spend most of its merges memorizing a handful of my
own catchphrases, and char-level keeps the vocab small (a few hundred) which
makes the output projection cheap.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List


class CharTokenizer:
    """Maps single characters <-> integer ids.

    Attributes:
        itos: list where itos[i] is the character for id i.
        stoi: inverse mapping, character -> id.
    """

    def __init__(self, chars: Iterable[str]):
        # Sorted so the vocab is deterministic across runs/machines: the same
        # corpus always produces the same id assignment, which matters because
        # checkpoints store ids, not characters.
        self.itos: List[str] = sorted(set(chars))
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    # ---- construction -----------------------------------------------------

    @classmethod
    def from_text(cls, text: str, min_freq: int = 1) -> "CharTokenizer":
        """Build a vocab from raw text.

        min_freq drops characters seen fewer than min_freq times. On a Discord
        corpus this prunes one-off emoji and stray CJK that the model could
        never learn anything useful about -- they'd just be dead embedding rows.
        """
        counts = Counter(text)
        chars = [ch for ch, n in counts.items() if n >= min_freq]
        if not chars:
            raise ValueError(f"No characters survived min_freq={min_freq}")
        return cls(chars)

    @classmethod
    def from_file(cls, path: str | Path, min_freq: int = 1) -> "CharTokenizer":
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_text(text, min_freq=min_freq)

    # ---- core API ---------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, s: str, drop_unknown: bool = True) -> List[int]:
        """String -> list of token ids.

        There is no <unk> token. Characters outside the vocab are dropped by
        default (relevant when a generation prompt contains something the
        corpus never had). Pass drop_unknown=False to fail loudly instead.
        """
        if drop_unknown:
            return [self.stoi[c] for c in s if c in self.stoi]
        try:
            return [self.stoi[c] for c in s]
        except KeyError as e:
            raise KeyError(f"Character {e.args[0]!r} is not in the vocabulary") from None

    def decode(self, ids: Iterable[int]) -> str:
        """List of token ids -> string."""
        return "".join(self.itos[i] for i in ids)

    # ---- persistence ------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the vocab as JSON so generation can reload the exact mapping."""
        Path(path).write_text(
            json.dumps({"itos": self.itos}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls.__new__(cls)  # bypass __init__ so we keep the saved ordering
        tok.itos = list(data["itos"])
        tok.stoi = {ch: i for i, ch in enumerate(tok.itos)}
        return tok

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"
