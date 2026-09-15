"""
A GPT-style decoder-only transformer, written from scratch.

No nn.MultiheadAttention, no nn.TransformerEncoderLayer -- attention is explicit
Q/K/V projections plus a manual scaled dot-product, because the whole point of
this project is being able to explain every tensor operation in the stack.

Shapes use these names throughout:
    B  = batch size
    T  = sequence length (<= config.block_size)
    C  = d_model (embedding dim)
    nh = n_heads
    hd = head_dim = C // nh
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 6
    block_size: int = 256      # max context length the positional embedding covers
    dropout: float = 0.1
    tie_weights: bool = False  # see GPT.__init__ for why this defaults off here

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention.

    Q, K and V are three separate Linear layers rather than one fused
    3*d_model projection. The fused version is what production code does (one
    GEMM instead of three), but keeping them separate makes the KV-cache work in
    Phase 2 read more clearly -- you can see exactly which projections are the
    ones you get to skip recomputing.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Lower-triangular mask, registered as a buffer so it moves with .to(device)
        # but is NOT a learnable parameter. persistent=False keeps it out of the
        # checkpoint, since it is a pure constant we can always rebuild.
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(config.block_size, config.block_size, dtype=torch.bool)
            ).view(1, 1, config.block_size, config.block_size),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        """
        x:       (B, T, C) -- during cached generation T is usually 1
        past_kv: (past_k, past_v), each (B, nh, P, hd), for the P tokens already
                 processed. None means "no history", i.e. the training path.
        returns: y (B, T, C), present (k, v) each (B, nh, P+T, hd) or None
        """
        B, T, C = x.shape
        nh, hd = self.n_heads, self.head_dim

        # --- project to Q/K/V -------------------------------------------------
        # (B, T, C) -> split C into (nh, hd) -> put heads on dim 1 so each head is
        # an independent (T, hd) sequence for the matmuls below.
        # Note we only ever project the T *new* tokens. That is the entire saving:
        # the K/V for the P cached tokens were computed on earlier steps and their
        # values cannot change, because attention is causal -- nothing that arrives
        # later can alter what an earlier position projected to.
        q = self.q_proj(x).view(B, T, nh, hd).transpose(1, 2)   # (B, nh, T, hd)
        k = self.k_proj(x).view(B, T, nh, hd).transpose(1, 2)   # (B, nh, T, hd)
        v = self.v_proj(x).view(B, T, nh, hd).transpose(1, 2)   # (B, nh, T, hd)

        # --- splice in the cache ----------------------------------------------
        past_len = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.size(2)
            # Concatenate along the sequence dim: (B,nh,P,hd) + (B,nh,T,hd)
            #                                  -> (B,nh,P+T,hd)
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        S = k.size(2)                                           # S = P + T
        assert S == past_len + T

        # --- scaled dot-product ----------------------------------------------
        # (B, nh, T, hd) @ (B, nh, hd, S) -> (B, nh, T, S)
        # Note this is now RECTANGULAR: T queries (the new tokens only) against S
        # keys (everything). With T=1 this is a matrix-vector product instead of
        # the T x T matrix-matrix product the uncached path does.
        # The 1/sqrt(hd) scaling keeps dot products from growing with head_dim,
        # which would otherwise push softmax into a saturated, near-one-hot regime
        # where gradients vanish.
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hd))
        assert att.shape == (B, nh, T, S)

        # Mask BEFORE softmax: -inf entries become exactly 0 probability, so the
        # normalization only ever runs over allowed (past + current) positions.
        # Masking after softmax would leave the rows not summing to 1.
        #
        # The row offset is the subtle part. Query i here sits at ABSOLUTE position
        # past_len + i, so it may attend to keys 0 .. past_len+i. That is rows
        # past_len : past_len+T of the causal mask, columns : S. With past_len=0
        # this reduces to the familiar [:T, :T] triangle.
        #
        # For T=1 the slice is a single all-True row (a new token sees all history),
        # so masking is a no-op -- but the general form is what makes Phase 3 work,
        # where we push k draft tokens through at once on top of a non-empty cache
        # and genuinely need the triangle.
        att = att.masked_fill(
            self.causal_mask[:, :, past_len : past_len + T, :S] == 0, float("-inf")
        )
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # --- weighted sum of values -------------------------------------------
        y = att @ v                                             # (B, nh, T, hd)
        assert y.shape == (B, nh, T, hd)

        # Merge heads back: (B, nh, T, hd) -> (B, T, nh, hd) -> (B, T, C).
        # .contiguous() is required because transpose leaves a non-contiguous view
        # that .view() cannot reinterpret.
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.out_proj(y))

        return y, ((k, v) if use_cache else None)


class FeedForward(nn.Module):
    """Position-wise MLP: expand 4x, GELU, project back."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden = 4 * config.d_model
        self.fc = nn.Linear(config.d_model, hidden)
        self.proj = nn.Linear(hidden, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, C) -> (B, T, 4C) -> (B, T, C)
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-LN transformer block.

    Pre-LN (norm inside the residual branch) rather than the original post-LN:
    it leaves a clean identity path from input to output, so gradients reach the
    early layers without a warmup schedule babysitting them. Post-LN at this
    depth needs careful warmup to not diverge.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        # Only attention needs the cache. The FFN and both LayerNorms are
        # position-wise -- they touch one token at a time and have no cross-token
        # state, so feeding them 1 token instead of T is automatically correct.
        attn_out, present = self.attn(self.ln1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, present


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        # Learned absolute positional embeddings (as in GPT-2), not sinusoidal and
        # not RoPE. Learned is the simplest thing that works, and it keeps the
        # Phase 2 KV cache trivial: position t is just a table lookup at index t,
        # so feeding a single new token needs no recomputation.
        # Cost: the model cannot extrapolate past block_size at all.
        self.pos_emb = nn.Embedding(config.block_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying defaults OFF here, the opposite of GPT-2, and the reason is
        # the char-level vocab. Tying saves vocab_size * d_model params; at
        # vocab=50257 that is ~30% of a small model and a huge win, but at
        # vocab~126 it is ~48K params out of ~11M (0.4%) -- so all it buys is a
        # constraint forcing the input and output spaces to coincide. Flip it on
        # via config to A/B it.
        if config.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # Scaled init for the projections that write into the residual stream.
        # With n_layers residual adds, the stream variance grows ~linearly with
        # depth; shrinking these by 1/sqrt(2*n_layers) keeps activations in range
        # at init (GPT-2 trick; the 2 is because each block writes twice).
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("ffn.proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_kvs: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ):
        """
        idx:      (B, T) int64 token ids -- only the NEW tokens when past_kvs is given
        targets:  (B, T) int64 token ids, already shifted by one by the caller
        past_kvs: per-layer [(k, v), ...], each (B, nh, P, hd), or None
        returns:  logits (B, T, vocab_size), loss (scalar or None), present_kvs

        present_kvs is None unless use_cache=True. The return arity is fixed at 3
        rather than varying with use_cache, so call sites never have to branch on
        the shape of what they get back.
        """
        B, T = idx.shape

        # How many tokens are already in the cache. Read off the K tensor's
        # sequence dim, so the cache is self-describing and callers do not have to
        # track a separate length counter that could drift out of sync with it.
        past_len = past_kvs[0][0].size(2) if past_kvs is not None else 0

        # The bound is on past + new, not just new. Learned positional embeddings
        # only exist for indices < block_size, so this is a hard architectural
        # limit, not a tunable. The naive path handles overflow by cropping the
        # context and recomputing; a cache cannot do that, because cropping
        # renumbers every position and so invalidates every cached K/V at once.
        # See kv_cache.generate_cached for how that ceiling is handled.
        assert past_len + T <= self.config.block_size, (
            f"past ({past_len}) + new ({T}) = {past_len + T} exceeds "
            f"block_size {self.config.block_size}"
        )

        # Absolute positions of the new tokens: past_len .. past_len+T-1.
        # This is the other half of what makes caching correct. Feeding one token
        # with pos=0 instead of pos=past_len would embed it as if it were the start
        # of the sequence -- a bug that produces plausible-looking garbage rather
        # than an error.
        pos = torch.arange(past_len, past_len + T, device=idx.device)   # (T,)
        x = self.tok_emb(idx) + self.pos_emb(pos)               # (B,T,C) + (T,C) broadcasts
        x = self.drop(x)

        present_kvs = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            x, present = block(
                x,
                past_kv=past_kvs[i] if past_kvs is not None else None,
                use_cache=use_cache,
            )                                                   # (B, T, C)
            if use_cache:
                present_kvs.append(present)

        x = self.ln_f(x)
        logits = self.lm_head(x)                                # (B, T, vocab_size)
        assert logits.shape == (B, T, self.config.vocab_size)

        loss = None
        if targets is not None:
            # Flatten batch and time: cross_entropy wants (N, classes).
            loss = F.cross_entropy(
                logits.view(B * T, self.config.vocab_size),
                targets.reshape(B * T),
            )
        return logits, loss, present_kvs
