"""Feature clamping: set SAE features to a fixed value inside the forward pass and measure the effect.

The causal check on feature attribution. At the SAE's hook point the
residual stream is edited so the chosen features read `value` instead of
what they would have read:

    resid' = resid + Σ_f (value - act_f) · W_dec[f]

Only those features' decoder contributions change; the SAE's reconstruction
error stays in the stream, so an imperfect SAE doesn't count as an effect.
value = 0 switches features off (ablation); a large value switches them on
(steering).

Each feature is clamped on its own and then all together, against random
features with the same mean activation as a baseline.

Answers: "Does the output actually depend on these features?"
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from transformer_lens import HookedTransformer

matplotlib.use("Agg")

from .._helpers import tokens_to_ids
from ..feature_extraction.sae import hook_name
from ..metrics import last_positions, logit_diff
from ..result import InterpResult


def compute(
    model: HookedTransformer,
    sae,
    prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    features: list[int],
    output_dir: Path,
    value: float = 0.0,
    n_random: int = 5,
    seed: int = 0,
) -> InterpResult:
    """Clamp each listed feature, then all of them, and record the logit diff.

    Args:
        model: Loaded HookedTransformer.
        sae: SAE on one of the model's hook points.
        prompts: Prompts the behaviour shows up on.
        positive_tokens: Correct answer token per prompt.
        negative_tokens: Distractor token per prompt.
        features: Feature indices to clamp.
        output_dir: Directory to save the plot.
        value: What to clamp them to; 0 = off.
        n_random: Random same-size feature sets clamped as a baseline, drawn
            from features active on these prompts.
        seed: Seed for the random sets.

    Returns:
        InterpResult with per-feature and joint logit diff change, and the
        random baseline for the joint change.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens = model.to_tokens(prompts)
    pos = last_positions(model, prompts)
    pos_ids = tokens_to_ids(model, positive_tokens)
    neg_ids = tokens_to_ids(model, negative_tokens)
    name = hook_name(sae)
    W_dec = sae.W_dec

    def run(idx: list[int]) -> float:
        def hook_fn(resid, hook):
            if not idx:
                return resid
            acts = sae.encode(resid.to(W_dec.dtype))[..., idx]          # [b, s, k]
            delta = (value - acts) @ W_dec[idx]                          # [b, s, d_model]
            return resid + delta.to(resid.dtype)
        with torch.no_grad(), model.hooks(fwd_hooks=[(name, hook_fn)]):
            return logit_diff(model(tokens), pos_ids, neg_ids, pos).mean().item()

    baseline = run([])
    per_feature = {str(f): round(run([f]) - baseline, 4) for f in features}
    joint = round(run(list(features)) - baseline, 4)

    # Random baseline: features that are active on these prompts, so the
    # comparison is "these features" vs "other features that are present".
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == name)
        active = (sae.encode(cache[name].to(W_dec.dtype)) > 0).any(dim=(0, 1)).nonzero().flatten()
    pool = [i for i in active.tolist() if i not in set(features)]
    gen = torch.Generator().manual_seed(seed)
    random_changes = []
    for _ in range(n_random if len(pool) >= len(features) else 0):
        pick = torch.randperm(len(pool), generator=gen)[: len(features)].tolist()
        random_changes.append(run([pool[i] for i in pick]) - baseline)
    random_mean = sum(random_changes) / len(random_changes) if random_changes else None

    # ── Plot ──────────────────────────────────────────────────────────────────
    labels = [f"#{f}" for f in features] + ["all together"]
    values = list(per_feature.values()) + [joint]
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(labels) + 2), 4))
    ax.bar(range(len(labels)), values, color=["#4575b4"] * len(features) + ["#d73027"])
    if random_mean is not None:
        ax.axhline(random_mean, color="grey", linestyle="--", label="random features, together")
        ax.legend(fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Change in logit diff")
    ax.set_title(f"Feature Clamping to {value:g} — baseline LD {baseline:.2f}")
    plt.tight_layout()
    plot_path = output_dir / "feature_clamping.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return InterpResult(
        plot_paths=[plot_path],
        data={
            "hook_name":          name,
            "value":              value,
            "baseline_ld":        round(baseline, 4),
            "per_feature_change": per_feature,
            "joint_change":       joint,
            "random_joint_change": None if random_mean is None else round(random_mean, 4),
        },
    )
