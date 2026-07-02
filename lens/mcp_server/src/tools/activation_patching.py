from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from transformer_lens import HookedTransformer

matplotlib.use("Agg")

from ..app.helpers import get_logit_diff, tokens_to_ids
from ..models.schemas import ExperimentResult
from ..config import PLOTS_DIR


def run_activation_patching(
    model: HookedTransformer,
    prompts: list[str],
    corrupted_prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Activation patching: for each layer × position, patch the residual stream
    from the clean run into the corrupted run and measure how much logit diff recovers.

    High recovery (→ 1.0) = that layer+position causally carries the IO information.
    Answers: "Which layer and token position carries the critical information?"

    Args:
        model: Loaded HookedTransformer.
        prompts: Clean prompts (model gets the right answer).
        corrupted_prompts: Corrupted prompts (model gets the wrong answer, e.g. names swapped).
        positive_tokens: Tokens to boost (e.g. [" Mary"] for IOI, [" she"] for gender bias).
        negative_tokens: Tokens to suppress (e.g. [" John"] for IOI, [" he"] for gender bias).
        output_dir: Directory to save the plot.

    Returns:
        ExperimentResult with layer × position recovery heatmap.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_tokens   = model.to_tokens(prompts)
    corrupt_tokens = model.to_tokens(corrupted_prompts)
    n_layers = model.cfg.n_layers
    seq_len  = clean_tokens.shape[1]
    io_ids   = tokens_to_ids(model, positive_tokens)
    s_ids    = tokens_to_ids(model, negative_tokens)

    clean_ld   = get_logit_diff(model, clean_tokens,   io_ids, s_ids)
    corrupt_ld = get_logit_diff(model, corrupt_tokens, io_ids, s_ids)
    total_diff = clean_ld - corrupt_ld
    print(
        f"  Clean LD: {clean_ld:.3f}  |  Corrupt LD: {corrupt_ld:.3f}  |  Delta: {total_diff:.3f}"
    )

    # Cache clean activations
    with torch.no_grad():
        _, clean_cache = model.run_with_cache(clean_tokens)

    patch_effects = torch.zeros(n_layers, seq_len)

    for layer in range(n_layers):
        clean_resid_layer = clean_cache["resid_post", layer]   # [batch, seq, d_model]

        for pos in range(seq_len):
            clean_act = clean_resid_layer[:, pos:pos + 1, :].clone()

            def hook_fn(value, hook, ca=clean_act, p=pos):
                value = value.clone()
                value[:, p:p + 1, :] = ca
                return value

            hook_name = f"blocks.{layer}.hook_resid_post"
            with torch.no_grad():
                with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
                    patched_ld = get_logit_diff(model, corrupt_tokens, io_ids, s_ids)

            # Normalised recovery: 0 = no recovery, 1 = full recovery
            if abs(total_diff) > 1e-6:
                patch_effects[layer, pos] = (patched_ld - corrupt_ld) / total_diff

        if layer % 3 == 0:
            print(f"  Layer {layer}/{n_layers - 1} done")

    token_strs = model.to_str_tokens(clean_tokens[0])

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(patch_effects.numpy(), cmap="RdBu", vmin=-0.5, vmax=1.0, aspect="auto")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(token_strs, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_layers))
    ax.set_title(
        f"Activation Patching — Normalised Logit Diff Recovery\n"
        f"1.0 = full recovery (critical), 0 = no effect"
    )
    plt.colorbar(im, ax=ax, label="Fraction recovered")
    plt.tight_layout()

    plot_path = output_dir / "activation_patching.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    flat    = patch_effects.flatten()
    top_idx = flat.topk(5).indices
    top_pos = [
        (int(i // seq_len), token_strs[int(i % seq_len)], flat[i].item())
        for i in top_idx
    ]

    return ExperimentResult(
        name="Activation Patching",
        tool="activation_patching",
        model_name=model.cfg.model_name,
        prompts=prompts,
        plot_paths=[plot_path],
        data={
            "clean_ld":                   round(clean_ld, 4),
            "corrupt_ld":                 round(corrupt_ld, 4),
            "patch_effects":              patch_effects.tolist(),
            "token_strs":                 token_strs,
            "top_recovery_positions":     [(f"L{l} tok='{t}'", round(v, 4)) for l, t, v in top_pos],
            "positive_tokens":            positive_tokens,
            "negative_tokens":            negative_tokens,
            "corrupted_prompts":          corrupted_prompts,
        },
        status="success",
    )
