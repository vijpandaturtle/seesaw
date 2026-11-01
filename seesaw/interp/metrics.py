"""The behaviour metrics every tool measures against, in one place.

Each tool used to compute its own logit diff; pathways compare numbers across
tools, so they need to be the same number. Everything here is differentiable
(attribution methods backprop through it) and per-prompt (callers decide how
to aggregate).

Positions: `model.to_tokens` right-pads a batch of unequal-length prompts, so
reading position -1 reads padding for the shorter ones. Tools that accept
prompts of different lengths read at `last_positions(...)` instead.
"""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer


def last_positions(model: HookedTransformer, prompts: list[str]) -> torch.Tensor:
    """Index of each prompt's final real token in a right-padded batch."""
    lengths = [model.to_tokens(p).shape[1] for p in prompts]
    return torch.tensor(lengths) - 1


def logit_diff(
    logits: torch.Tensor,
    positive_ids: list[int],
    negative_ids: list[int],
    pos: int | torch.Tensor = -1,
) -> torch.Tensor:
    """Per-prompt logit(positive) - logit(negative).

    Args:
        logits: [batch, seq, d_vocab].
        positive_ids: Correct-answer token id per prompt.
        negative_ids: Distractor token id per prompt.
        pos: Position to read: one int for every prompt, or a [batch] tensor
            of per-prompt positions (see `last_positions`).

    Returns:
        [batch] tensor, still attached to the graph.
    """
    rows = torch.arange(logits.shape[0], device=logits.device)
    if isinstance(pos, torch.Tensor):
        pos = pos.to(logits.device)
    final = logits[rows, pos]                                   # [batch, d_vocab]
    pos_ids = torch.tensor(positive_ids, device=logits.device)
    neg_ids = torch.tensor(negative_ids, device=logits.device)
    return final[rows, pos_ids] - final[rows, neg_ids]


def recovery(patched: float, corrupt: float, clean: float) -> float:
    """Fraction of the clean-vs-corrupt gap a patch restores: 0 = none, 1 = all."""
    gap = clean - corrupt
    return 0.0 if abs(gap) < 1e-6 else (patched - corrupt) / gap
