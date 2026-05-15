from pathlib import Path
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import torch
from transformer_lens import HookedTransformer

matplotlib.use("Agg")

from ..app.helpers import get_logit_diff, tokens_to_ids
from ..app.schemas import ExperimentResult
from ..config import PLOTS_DIR


def run_ablation(
    model: HookedTransformer,
    prompts: list[str],
    io_tokens: list[str],
    subject_tokens: list[str],
    ablation_type: Literal["zero", "mean"] = "mean",
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Ablate each attention head and measure the drop in logit diff.

    Hooks on hook_z (always available) rather than hook_result.
    Large positive drop = causally important head.

    Answers: "Is this component causally necessary for the behaviour?"

    Args:
        model: Loaded HookedTransformer.
        prompts: List of input prompt strings.
        io_tokens: Correct (indirect object) token strings.
        subject_tokens: Incorrect (subject) token strings.
        ablation_type: "zero" to zero out the head, "mean" to replace with mean activation.
        output_dir: Directory to save the plot.

    Returns:
        ExperimentResult with ablation heatmap and top important heads.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens   = model.to_tokens(prompts)
    n_layers = model.cfg.n_layers
    n_heads  = model.cfg.n_heads
    io_ids   = tokens_to_ids(model, io_tokens)
    s_ids    = tokens_to_ids(model, subject_tokens)

    baseline_ld = get_logit_diff(model, tokens, io_ids, s_ids)
    print(f"  Baseline logit diff: {baseline_ld:.4f}")

    # Precompute mean z activations for mean ablation
    mean_z: dict[int, torch.Tensor] = {}
    if ablation_type == "mean":
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        for layer in range(n_layers):
            # Mean over batch and seq → [1, 1, n_heads, d_head]
            mean_z[layer] = cache["z", layer].mean(dim=[0, 1], keepdim=True)

    ablation_effects = torch.zeros(n_layers, n_heads)

    for layer in range(n_layers):
        for head in range(n_heads):
            if ablation_type == "zero":
                def hook_fn(value, hook, h=head):
                    value = value.clone()
                    value[:, :, h, :] = 0.0
                    return value
            else:
                mz = mean_z[layer][:, :, head, :]   # [1, 1, d_head]

                def hook_fn(value, hook, h=head, mz=mz):
                    value = value.clone()
                    value[:, :, h, :] = mz
                    return value

            hook_name = f"blocks.{layer}.attn.hook_z"
            with torch.no_grad():
                with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
                    ablated_ld = get_logit_diff(model, tokens, io_ids, s_ids)

            ablation_effects[layer, head] = baseline_ld - ablated_ld

        if layer % 3 == 0:
            print(f"  Layer {layer}/{n_layers - 1} done")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    vmax = max(ablation_effects.abs().max().item(), 1e-6)
    im   = ax.imshow(ablation_effects.numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_layers))
    ax.set_title(
        f"{ablation_type.capitalize()} Ablation — Drop in Logit Diff\n"
        "Red = important (removing hurts), Blue = suppressive (removing helps)"
    )
    plt.colorbar(im, ax=ax, label="Logit diff drop")
    plt.tight_layout()

    plot_path = output_dir / f"ablation_{ablation_type}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    flat      = ablation_effects.flatten()
    top_idx   = flat.topk(5).indices
    top_heads = [(int(i // n_heads), int(i % n_heads), flat[i].item()) for i in top_idx]

    return ExperimentResult(
        name=f"Ablation ({ablation_type})",
        tool="ablation",
        model_name=model.cfg.model_name,
        prompts=prompts,
        plot_paths=[plot_path],
        data={
            "ablation_type":  ablation_type,
            "baseline_ld":    round(baseline_ld, 4),
            "effects":        ablation_effects.tolist(),
            "top_heads":      [(f"L{l}H{h}", round(v, 4)) for l, h, v in top_heads],
            "io_tokens":      io_tokens,
            "subject_tokens": subject_tokens,
        },
        status="success",
    )
