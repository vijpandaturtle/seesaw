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
    control_texts: list[str] | None = None,
    n_random: int = 3,
    seed: int = 0,
) -> InterpResult:
    """Sweep the steering strength and record the mean logit diff at each.

    Two controls run alongside: random directions of the same norm swept the
    same way (a real direction should move the behaviour far more), and the
    language-modelling loss at the strongest push (a big move with a big loss
    increase is damage, not steering).

    Args:
        model: Loaded HookedTransformer.
        prompts: Prompts to steer.
        positive_tokens: Token per prompt the direction should push toward.
        negative_tokens: Token per prompt it should push away from.
        direction: [d_model] vector to add, unnormalised.
        layer: Add it to blocks.{layer}.hook_resid_post.
        output_dir: Directory to save the plot.
        alphas: Multiples of `direction` to try.
        control_texts: Text to measure loss on. Defaults to `prompts`.
        n_random: Random directions to compare against.
        seed: Seed for the random directions.

    Returns:
        InterpResult with the logit diff at each alpha, the curve's slope, the
        random directions' mean absolute slope, and the loss before and at the
        strongest push.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens = model.to_tokens(prompts)
    pos = last_positions(model, prompts)
    pos_ids = tokens_to_ids(model, positive_tokens)
    neg_ids = tokens_to_ids(model, negative_tokens)
    vec = torch.tensor(direction, device=model.cfg.device, dtype=model.cfg.dtype)
    hook_name = f"blocks.{layer}.hook_resid_post"
    # One text at a time: a padded batch would score predictions of padding.
    control_tokens = [model.to_tokens(t) for t in (control_texts or prompts)]
    a = torch.tensor(list(alphas))

    def hooks(v, alpha):
        return [(hook_name, lambda value, hook: value + alpha * v)]

    def sweep(v) -> tuple[list[float], float]:
        lds = []
        for alpha in alphas:
            with torch.no_grad(), model.hooks(fwd_hooks=hooks(v, alpha)):
                lds.append(logit_diff(model(tokens), pos_ids, neg_ids, pos).mean().item())
        # Least-squares slope of LD against alpha: logit diff per unit of direction.
        y = torch.tensor(lds)
        slope = (((a - a.mean()) * (y - y.mean())).sum() / ((a - a.mean()) ** 2).sum()).item()
        return lds, slope

    def loss(alpha) -> float:
        with torch.no_grad(), model.hooks(fwd_hooks=hooks(vec, alpha)):
            losses = [model(t, return_type="loss").item() for t in control_tokens]
        return sum(losses) / len(losses)

    lds, slope = sweep(vec)

    gen = torch.Generator().manual_seed(seed)
    random_slopes = []
    for _ in range(n_random):
        r = torch.randn(vec.shape[0], generator=gen).to(vec.device, vec.dtype)
        random_slopes.append(abs(sweep(r * vec.norm() / r.norm())[1]))
    random_abs_slope = sum(random_slopes) / len(random_slopes) if random_slopes else None

    strongest = max(alphas, key=abs)
    loss_before, loss_after = loss(0.0), loss(strongest)
    monotonic = all(y2 >= y1 for y1, y2 in zip(lds, lds[1:])) or all(
        y2 <= y1 for y1, y2 in zip(lds, lds[1:]))

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(alphas), lds, marker="o", color="#d73027")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("alpha (multiples of the direction)")
    ax.set_ylabel(f"logit({positive_tokens[0]!r}) - logit({negative_tokens[0]!r})")
    ax.set_title(
        f"Steering at layer {layer} — slope {slope:+.2f}"
        + (f" (random dirs: ±{random_abs_slope:.2f})" if random_abs_slope is not None else "")
        + f"\nloss {loss_before:.2f} → {loss_after:.2f} at alpha {strongest:+g}")
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
            "random_abs_slope":   None if random_abs_slope is None else round(random_abs_slope, 4),
            "control_loss_before": round(loss_before, 4),
            "control_loss_after":  round(loss_after, 4),
            "control_loss_alpha":  strongest,
        },
    )
