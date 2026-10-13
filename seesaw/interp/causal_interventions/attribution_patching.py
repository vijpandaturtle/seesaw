"""Attribution patching: estimate every head's patching effect from one forward and one backward pass.

Activation patching needs one forward pass per component; on a 70B model
that's tens of thousands of passes. Attribution patching (Nanda 2023; Syed et
al. 2023) replaces each patch with its first-order Taylor estimate:

    effect(c) ≈ (clean_act_c - corrupt_act_c) · ∂metric/∂act_c   (at the corrupted run)

so the whole sweep costs one clean pass, one corrupted pass and one backward
pass. The estimate is linear, so it is wrong where the metric saturates — big
effects are usually ranked right but mis-sized. That's why it nominates and
`head_patching` verifies: never report an attribution-patching number as a
causal effect.

Scores are normalised like activation patching's recovery: 1.0 ≈ patching that
component alone would restore the full clean-vs-corrupt logit-diff gap.

Answers: "Which heads and MLPs are worth testing causally?"
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from transformer_lens import HookedTransformer

matplotlib.use("Agg")

from .._helpers import top_heads, tokens_to_ids
from ..metrics import logit_diff
from ..result import InterpResult


def _is_component(name: str) -> bool:
    return name.endswith(("hook_z", "hook_mlp_out"))


def compute(
    model: HookedTransformer,
    prompts: list[str],
    corrupted_prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    output_dir: Path,
    top_k: int = 10,
) -> InterpResult:
    """Estimate the patching effect of every attention head and MLP layer at once.

    Args:
        model: Loaded HookedTransformer.
        prompts: Clean prompts (model gets the right answer).
        corrupted_prompts: Same template with the key entity swapped; must
            tokenize to the same length as the clean prompts.
        positive_tokens: Correct answer token per prompt (e.g. " Mary").
        negative_tokens: Distractor token per prompt (e.g. " John").
        output_dir: Directory to save the plot.
        top_k: How many heads to report in top_heads.

    Returns:
        InterpResult with [layer, head] and [layer] estimated recovery.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_tokens   = model.to_tokens(prompts)
    corrupt_tokens = model.to_tokens(corrupted_prompts)
    if clean_tokens.shape != corrupt_tokens.shape:
        raise ValueError(
            f"clean and corrupted prompts tokenize to different shapes "
            f"{tuple(clean_tokens.shape)} vs {tuple(corrupt_tokens.shape)}; "
            f"attribution patching needs position-aligned pairs"
        )
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    pos_ids = tokens_to_ids(model, positive_tokens)
    neg_ids = tokens_to_ids(model, negative_tokens)

    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens, names_filter=_is_component)
    clean_ld = logit_diff(clean_logits, pos_ids, neg_ids).mean().item()

    # The corrupted pass keeps the graph so we can differentiate the metric
    # with respect to each component's output. The embedding is re-rooted as a
    # leaf that requires grad, so this works even when the model's parameters
    # have requires_grad=False.
    acts: dict[str, torch.Tensor] = {}

    def root(act, hook):
        return act.detach().requires_grad_(True)

    def keep(act, hook):
        acts[hook.name] = act
        return act

    hooks = [("hook_embed", root)] + [
        (name, keep) for name in clean_cache.keys()
    ]
    with torch.enable_grad():
        with model.hooks(fwd_hooks=hooks):
            corrupt_logits = model(corrupt_tokens)
        metric = logit_diff(corrupt_logits, pos_ids, neg_ids).mean()
        names = list(acts)
        grads = dict(zip(names, torch.autograd.grad(metric, [acts[n] for n in names])))
    corrupt_ld = metric.item()
    gap = clean_ld - corrupt_ld
    scale = gap if abs(gap) > 1e-6 else 1.0

    # The metric is already a batch mean, so each prompt's gradient carries a
    # 1/batch factor: summing over the batch (not averaging) gives the mean effect.
    head_effects = torch.zeros(n_layers, n_heads)
    mlp_effects  = torch.zeros(n_layers)
    for layer in range(n_layers):
        z_name, mlp_name = f"blocks.{layer}.attn.hook_z", f"blocks.{layer}.hook_mlp_out"
        delta_z = clean_cache[z_name] - acts[z_name].detach()           # [b, s, h, d_head]
        head_effects[layer] = (
            (delta_z * grads[z_name]).sum(dim=(0, 1, 3)).cpu() / scale
        )
        if mlp_name in grads:                                            # absent in attn-only models
            delta_mlp = clean_cache[mlp_name] - acts[mlp_name].detach()  # [b, s, d_model]
            mlp_effects[layer] = (delta_mlp * grads[mlp_name]).sum().cpu() / scale
    acts.clear()
    del clean_cache, grads

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [3, 1]})
    vmax = max(head_effects.abs().max().item(), 1e-6)
    im = axes[0].imshow(head_effects.numpy(), cmap="RdBu", vmin=-vmax, vmax=vmax)
    axes[0].set_xlabel("Head")
    axes[0].set_ylabel("Layer")
    axes[0].set_xticks(range(n_heads))
    axes[0].set_yticks(range(n_layers))
    axes[0].set_title("Estimated recovery per head")
    plt.colorbar(im, ax=axes[0])
    colors = ["#d73027" if v > 0 else "#4575b4" for v in mlp_effects.tolist()]
    axes[1].barh(range(n_layers), mlp_effects.numpy(), color=colors)
    axes[1].set_yticks(range(n_layers))
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("Estimated recovery per MLP")
    plt.suptitle(
        "Attribution Patching — linear estimate, verify before trusting\n"
        f"clean LD {clean_ld:.2f}, corrupt LD {corrupt_ld:.2f}"
    )
    plt.tight_layout()
    plot_path = output_dir / "attribution_patching.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return InterpResult(
        plot_paths=[plot_path],
        data={
            "clean_ld":        round(clean_ld, 4),
            "corrupt_ld":      round(corrupt_ld, 4),
            "head_effects":    head_effects.tolist(),
            "mlp_effects":     mlp_effects.tolist(),
            "top_heads":       top_heads(head_effects, top_k),
            "positive_tokens": positive_tokens,
            "negative_tokens": negative_tokens,
            "corrupted_prompts": corrupted_prompts,
        },
    )
