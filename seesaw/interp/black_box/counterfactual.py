"""Counterfactual evaluation: swap one attribute in the input and measure how much the output moves.

Inputs and outputs only — no hooks, no caches — so it runs on any model,
including ones whose internals you can't reach. That makes it the measuring
step of an audit: if swapping the attribute doesn't move the output, there's
nothing for the mechanistic tools to explain.

Each prompt is paired with a counterfactual that differs only in the
attribute (" The nurse said that" vs " The engineer said that"). Optional
control prompts carry no attribute; their logit diff shows the model's
baseline preference, so an effect isn't mistaken for a constant bias.

Answers: "Does this attribute change the model's behaviour, and how consistently?"
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from transformer_lens import HookedTransformer

matplotlib.use("Agg")

from .._helpers import tokens_to_ids
from ..metrics import last_positions, logit_diff
from ..result import InterpResult


def _lds(model, prompts, pos_ids, neg_ids) -> torch.Tensor:
    with torch.no_grad():
        logits = model(model.to_tokens(prompts))
    return logit_diff(logits, pos_ids, neg_ids, last_positions(model, prompts)).cpu()


def compute(
    model: HookedTransformer,
    prompts: list[str],
    counterfactual_prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    output_dir: Path,
    control_prompts: list[str] | None = None,
) -> InterpResult:
    """Measure the logit-diff change between each prompt and its counterfactual.

    Args:
        model: Loaded HookedTransformer.
        prompts: Prompts carrying the attribute (e.g. female-stereotyped occupations).
        counterfactual_prompts: Same prompts with the attribute swapped, one per prompt.
        positive_tokens: Token the attribute should push toward (e.g. " she"), per prompt.
        negative_tokens: Token it should push away from (e.g. " he"), per prompt.
        output_dir: Directory to save the plot.
        control_prompts: Optional attribute-free prompts (e.g. "The person said that").

    Returns:
        InterpResult with per-pair effects, their mean and how consistent the sign is.
    """
    if len(prompts) != len(counterfactual_prompts):
        raise ValueError("need exactly one counterfactual per prompt")
    output_dir.mkdir(parents=True, exist_ok=True)
    pos_ids = tokens_to_ids(model, positive_tokens)
    neg_ids = tokens_to_ids(model, negative_tokens)

    ld   = _lds(model, prompts, pos_ids, neg_ids)
    cf   = _lds(model, counterfactual_prompts, pos_ids, neg_ids)
    gaps = ld - cf
    mean_gap = gaps.mean().item()
    consistency = (torch.sign(gaps) == (1 if mean_gap >= 0 else -1)).float().mean().item()

    control = None
    if control_prompts:
        n = len(control_prompts)
        control = _lds(model, control_prompts, pos_ids[:1] * n, neg_ids[:1] * n)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(prompts) + 2), 4))
    x = torch.arange(len(prompts))
    ax.bar(x - 0.2, ld.numpy(), width=0.4, label="prompt", color="#d73027")
    ax.bar(x + 0.2, cf.numpy(), width=0.4, label="counterfactual", color="#4575b4")
    if control is not None:
        ax.axhline(control.mean().item(), color="grey", linestyle="--", label="control mean")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x.tolist())
    ax.set_xticklabels([p[-28:] for p in prompts], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(f"logit({positive_tokens[0]!r}) - logit({negative_tokens[0]!r})")
    ax.set_title(f"Counterfactual Evaluation — mean effect {mean_gap:+.2f}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plot_path = output_dir / "counterfactual.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return InterpResult(
        plot_paths=[plot_path],
        data={
            "prompt_lds":         [round(v, 4) for v in ld.tolist()],
            "counterfactual_lds": [round(v, 4) for v in cf.tolist()],
            "effects":            [round(v, 4) for v in gaps.tolist()],
            "mean_effect":        round(mean_gap, 4),
            "sign_consistency":   round(consistency, 4),
            "control_mean_ld":    None if control is None else round(control.mean().item(), 4),
            "counterfactual_prompts": counterfactual_prompts,
            "positive_tokens":    positive_tokens,
            "negative_tokens":    negative_tokens,
        },
    )
