"""Directional ablation: project a direction out of the residual stream everywhere and see what breaks.

The "does removing it fix it" test (Arditi et al. 2024; LEACE is the
provably-minimal version). The direction is removed from the residual stream
entering every block and leaving the last one, at every position, so no layer
can read it.

Two comparisons keep the result honest:
- random directions of the same dimension, removed the same way: if they
  shrink the effect as much, the direction isn't special;
- language-modelling loss on control text: removal that wrecks the model
  "removes" every behaviour, which says nothing about this one.

If `counterfactual_prompts` are given the measured effect is the gap between
prompts and counterfactuals (the audit question: does the attribute still
change the output?); otherwise it's the mean logit diff on `prompts`.

Answers: "Does the model need this direction for the behaviour?"
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


def _hooks(model, unit: torch.Tensor | None):
    if unit is None:
        return []

    def project_out(value, hook):
        return value - (value @ unit).unsqueeze(-1) * unit

    names = [f"blocks.{l}.hook_resid_pre" for l in range(model.cfg.n_layers)]
    names.append(f"blocks.{model.cfg.n_layers - 1}.hook_resid_post")
    return [(n, project_out) for n in names]


def compute(
    model: HookedTransformer,
    prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    direction: list[float],
    output_dir: Path,
    counterfactual_prompts: list[str] | None = None,
    control_texts: list[str] | None = None,
    n_random: int = 5,
    seed: int = 0,
) -> InterpResult:
    """Remove a direction from the residual stream and measure the behaviour before and after.

    Args:
        model: Loaded HookedTransformer.
        prompts: Prompts the behaviour shows up on.
        positive_tokens: Token per prompt the behaviour favours.
        negative_tokens: Token per prompt it disfavours.
        direction: [d_model] direction to remove (normalised internally).
        output_dir: Directory to save the plot.
        counterfactual_prompts: Optional attribute-swapped prompts; when given,
            the effect measured is LD(prompts) - LD(counterfactuals).
        control_texts: Text to measure language-modelling loss on, as a check
            that the edit didn't damage the model overall. Defaults to `prompts`.
        n_random: Random directions to compare against.
        seed: Seed for the random directions.

    Returns:
        InterpResult with the effect before/after, fraction removed, the random
        baseline, and the loss change on control text.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pos_ids = tokens_to_ids(model, positive_tokens)
    neg_ids = tokens_to_ids(model, negative_tokens)
    tokens = model.to_tokens(prompts)
    pos = last_positions(model, prompts)
    cf_tokens = model.to_tokens(counterfactual_prompts) if counterfactual_prompts else None
    cf_pos = last_positions(model, counterfactual_prompts) if counterfactual_prompts else None
    # One text at a time: a padded batch would score predictions of padding.
    control_tokens = [model.to_tokens(t) for t in (control_texts or prompts)]
    dev, dt = model.cfg.device, model.cfg.dtype

    def effect(unit) -> float:
        with torch.no_grad(), model.hooks(fwd_hooks=_hooks(model, unit)):
            ld = logit_diff(model(tokens), pos_ids, neg_ids, pos).mean().item()
            if cf_tokens is None:
                return ld
            return ld - logit_diff(model(cf_tokens), pos_ids, neg_ids, cf_pos).mean().item()

    def loss(unit) -> float:
        with torch.no_grad(), model.hooks(fwd_hooks=_hooks(model, unit)):
            losses = [model(t, return_type="loss").item() for t in control_tokens]
        return sum(losses) / len(losses)

    vec = torch.tensor(direction, device=dev, dtype=dt)
    unit = vec / vec.norm()
    before, after = effect(None), effect(unit)
    removed = 1 - after / before if abs(before) > 1e-6 else 0.0

    gen = torch.Generator().manual_seed(seed)
    random_removed = []
    for _ in range(n_random):
        r = torch.randn(vec.shape[0], generator=gen).to(dev, dt)
        random_removed.append(1 - effect(r / r.norm()) / before if abs(before) > 1e-6 else 0.0)
    random_mean = sum(random_removed) / len(random_removed) if random_removed else 0.0

    loss_before, loss_after = loss(None), loss(unit)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([0, 1], [before, after], color=["grey", "#d73027"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["original", "direction removed"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("effect (logit diff" + (" gap)" if cf_tokens is not None else ")"))
    ax.set_title(
        f"Directional Ablation — {removed:.0%} removed "
        f"(random dirs: {random_mean:.0%})\nloss on control {loss_before:.2f} → {loss_after:.2f}"
    )
    plt.tight_layout()
    plot_path = output_dir / "directional_ablation.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return InterpResult(
        plot_paths=[plot_path],
        data={
            "effect_before":          round(before, 4),
            "effect_after":           round(after, 4),
            "fraction_removed":       round(removed, 4),
            "random_fraction_removed": round(random_mean, 4),
            "control_loss_before":    round(loss_before, 4),
            "control_loss_after":     round(loss_after, 4),
            "measured":               "gap vs counterfactuals" if cf_tokens is not None else "logit diff",
        },
    )
