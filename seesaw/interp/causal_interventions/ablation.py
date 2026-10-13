"""Ablation: silence each attention head and measure the drop in logit diff.

Hooks on hook_z (always available) rather than hook_result. Large positive
drop = causally important head.

Three ways to silence a head:
- zero: set its output to 0. Pushes the model off-distribution; overstates effects.
- mean: replace with its mean over the prompts. The default.
- resample: replace with its output on a different prompt (`resample_prompts`,
  e.g. the corrupted set). Keeps activations on-distribution, which matters
  more on large models where mean activations are less representative.

`heads` restricts the sweep to a shortlist, so verification after a cheap
nominating pass costs one batch per layer touched instead of the whole model.

Answers: "Is this component causally necessary for the behaviour?"
"""

from pathlib import Path
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import torch
from transformer_lens import HookedTransformer

matplotlib.use("Agg")

from .._helpers import (
    get_logit_diff, parse_head, row_logit_diffs, sweep_chunks, tokens_to_ids,
)
from ..result import InterpResult


def compute(
    model: HookedTransformer,
    prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    output_dir: Path,
    ablation_type: Literal["zero", "mean", "resample"] = "mean",
    heads: list[str] | None = None,
    resample_prompts: list[str] | None = None,
) -> InterpResult:
    """Ablate each attention head and measure the drop in logit diff.

    Args:
        model: Loaded HookedTransformer.
        prompts: List of input prompt strings.
        positive_tokens: Tokens to boost (e.g. [" Mary"] for IOI, [" she"] for gender bias).
        negative_tokens: Tokens to suppress (e.g. [" John"] for IOI, [" he"] for gender bias).
        output_dir: Directory to save the plot.
        ablation_type: "zero" to zero out the head, "mean" to replace with mean
            activation, "resample" to replace with the head's output on
            `resample_prompts`.
        heads: Only ablate these heads (e.g. ["L9H9", "L10H7"]); None = every head.
            Heads not ablated score 0 in `effects`.
        resample_prompts: One per prompt, tokenizing to the same length;
            required for "resample".

    Returns:
        InterpResult with ablation heatmap and top important heads.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens   = model.to_tokens(prompts)
    n_layers = model.cfg.n_layers
    n_heads  = model.cfg.n_heads
    io_ids   = tokens_to_ids(model, positive_tokens)
    s_ids    = tokens_to_ids(model, negative_tokens)

    baseline_ld = get_logit_diff(model, tokens, io_ids, s_ids)
    print(f"  Baseline logit diff: {baseline_ld:.4f}")

    # Precompute mean z activations for mean ablation. Only hook_z is read, so
    # caching anything else is wasted memory — on a large model with a long
    # prompt the attention patterns alone are what exhaust it.
    mean_z: dict[int, torch.Tensor] = {}
    resample_z: dict[int, torch.Tensor] = {}
    if ablation_type == "mean":
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens, names_filter=lambda name: name.endswith("hook_z")
            )
        for layer in range(n_layers):
            # Mean over batch and seq → [1, 1, n_heads, d_head]
            mean_z[layer] = cache["z", layer].mean(dim=[0, 1], keepdim=True)
        del cache
    elif ablation_type == "resample":
        if not resample_prompts:
            raise ValueError("ablation_type='resample' needs resample_prompts")
        resample_tokens = model.to_tokens(resample_prompts)
        if resample_tokens.shape != tokens.shape:
            raise ValueError(
                f"resample_prompts tokenize to {tuple(resample_tokens.shape)}, "
                f"prompts to {tuple(tokens.shape)}; they must be position-aligned"
            )
        with torch.no_grad():
            _, cache = model.run_with_cache(
                resample_tokens, names_filter=lambda name: name.endswith("hook_z")
            )
        for layer in range(n_layers):
            resample_z[layer] = cache["z", layer]            # [batch, seq, n_heads, d_head]
        del cache

    # Which heads to sweep, per layer. Default: all of them.
    targets: dict[int, list[int]] = {}
    if heads is None:
        targets = {layer: list(range(n_heads)) for layer in range(n_layers)}
    else:
        for name in heads:
            layer, head = parse_head(name)
            targets.setdefault(layer, []).append(head)

    ablation_effects = torch.zeros(n_layers, n_heads)
    n_prompts = tokens.shape[0]

    # Heads within a layer are independent interventions on the same prompts,
    # so they go in one batch instead of one forward pass each: the prompts are
    # tiled once per head and each slice has a different head ablated.
    for layer, layer_heads in sorted(targets.items()):
        hook_name = f"blocks.{layer}.attn.hook_z"

        for start, stop in sweep_chunks(len(layer_heads), n_prompts):
            chunk = torch.tensor(layer_heads[start:stop])
            batch = tokens.repeat(len(chunk), 1)          # head-major rows
            rows = torch.arange(batch.shape[0])
            head_of_row = chunk.to(batch.device)[rows // n_prompts]

            if ablation_type == "zero":
                def hook_fn(value, hook, r=rows, h=head_of_row):
                    value = value.clone()
                    value[r, :, h, :] = 0.0
                    return value
            elif ablation_type == "resample":
                # Row r takes its own prompt's counterpart from the resample run.
                replacement = resample_z[layer][rows % n_prompts, :, head_of_row, :]

                def hook_fn(value, hook, r=rows, h=head_of_row, rep=replacement):
                    value = value.clone()
                    value[r, :, h, :] = rep
                    return value
            else:
                # [len(heads), d_head] → one replacement vector per row,
                # broadcast across the sequence.
                replacement = mean_z[layer][0, 0][head_of_row].unsqueeze(1)

                def hook_fn(value, hook, r=rows, h=head_of_row, rep=replacement):
                    value = value.clone()
                    value[r, :, h, :] = rep
                    return value

            with torch.no_grad():
                with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
                    logits = model(batch)

            diffs = row_logit_diffs(logits, io_ids, s_ids, n_prompts)
            ablated_ld = diffs.view(len(chunk), n_prompts).mean(dim=1)
            ablation_effects[layer, chunk] = baseline_ld - ablated_ld.cpu()
            del logits

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

    return InterpResult(
        plot_paths=[plot_path],
        data={
            "ablation_type":  ablation_type,
            "ablated_heads":  heads,
            "baseline_ld":    round(baseline_ld, 4),
            "effects":        ablation_effects.tolist(),
            "top_heads":      [(f"L{l}H{h}", round(v, 4)) for l, h, v in top_heads],
            "positive_tokens": positive_tokens,
            "negative_tokens": negative_tokens,
        },
    )
