"""Steering vector: add a direction to the residual stream and watch the behaviour move.

The direction is usually a difference of means between two prompt sets (the
`direction` that `probing.linear_probe` returns, or contrastive activation
addition's pairs). It's added at every position of one layer's residual
stream, scaled by each value in `alphas`: alpha = 1 adds the full A-minus-B
difference, alpha = -1 subtracts it.

A dose-response curve — behaviour moving smoothly and monotonically with
alpha — is much stronger evidence that the direction controls the behaviour
than any single alpha.

Answers: "Does pushing along this direction change the behaviour, and by how much?"
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


def compute(
    model: HookedTransformer,
    prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    direction: list[float],
    layer: int,
    output_dir: Path,
    alphas: list[float] = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0),
) -> InterpResult:
    """Sweep the steering strength and record the mean logit diff at each.

    Args:
        model: Loaded HookedTransformer.
        prompts: Prompts to steer.
        positive_tokens: Token per prompt the direction should push toward.
        negative_tokens: Token per prompt it should push away from.
        direction: [d_model] vector to add, unnormalised.
        layer: Add it to blocks.{layer}.hook_resid_post.
        output_dir: Directory to save the plot.
        alphas: Multiples of `direction` to try.

    Returns:
        InterpResult with the logit diff at each alpha and the curve's slope.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens = model.to_tokens(prompts)
    pos = last_positions(model, prompts)
    pos_ids = tokens_to_ids(model, positive_tokens)
    neg_ids = tokens_to_ids(model, negative_tokens)
    vec = torch.tensor(direction, device=model.cfg.device, dtype=model.cfg.dtype)
    hook_name = f"blocks.{layer}.hook_resid_post"

    lds = []
    for alpha in alphas:
        def hook_fn(value, hook, a=alpha):
            return value + a * vec
        with torch.no_grad(), model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
            logits = model(tokens)
        lds.append(logit_diff(logits, pos_ids, neg_ids, pos).mean().item())

    # Least-squares slope of LD against alpha: logit diff per unit of direction.
    a = torch.tensor(list(alphas))
    y = torch.tensor(lds)
    slope = (((a - a.mean()) * (y - y.mean())).sum() / ((a - a.mean()) ** 2).sum()).item()
    monotonic = all(y2 >= y1 for y1, y2 in zip(lds, lds[1:])) or all(
        y2 <= y1 for y1, y2 in zip(lds, lds[1:]))

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(alphas), lds, marker="o", color="#d73027")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("alpha (multiples of the direction)")
    ax.set_ylabel(f"logit({positive_tokens[0]!r}) - logit({negative_tokens[0]!r})")
    ax.set_title(f"Steering at layer {layer} — slope {slope:+.2f}")
    plt.tight_layout()
    plot_path = output_dir / f"steering_L{layer}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return InterpResult(
        plot_paths=[plot_path],
        data={
            "layer":     layer,
            "alphas":    list(alphas),
            "lds":       [round(v, 4) for v in lds],
            "slope":     round(slope, 4),
            "monotonic": monotonic,
        },
    )
