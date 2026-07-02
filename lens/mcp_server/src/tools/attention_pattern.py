from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformer_lens import HookedTransformer

matplotlib.use("Agg")

from ..models.schemas import ExperimentResult
from ..config import PLOTS_DIR


def run_attention_pattern(
    model: HookedTransformer,
    prompts: list[str],
    layers: list[int] | None = None,
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Visualise attention weights per head per layer.

    Answers: "What does each head attend to?"

    Args:
        model: Loaded HookedTransformer.
        prompts: List of input prompt strings.
        layers: Specific layers to visualise. Defaults to 5 evenly-spaced layers.
        output_dir: Directory to save plots.

    Returns:
        ExperimentResult with one plot per layer.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens     = model.to_tokens(prompts)
    token_strs = model.to_str_tokens(tokens[0])
    seq_len    = tokens.shape[1]
    n_heads    = model.cfg.n_heads
    n_layers   = model.cfg.n_layers

    if layers is None:
        step   = max(1, n_layers // 5)
        layers = list(range(0, n_layers, step))[:5]

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    plot_paths, layer_data = [], {}

    for layer in layers:
        attn = cache["pattern", layer][0]   # [n_heads, seq, seq]
        layer_data[f"layer_{layer}"] = {
            "max_attn_per_head": attn[:, -1, :].max(-1).values.tolist(),
            "final_token_attn":  attn[:, -1, :].tolist(),
        }

        n_cols   = min(4, n_heads)
        n_rows   = (n_heads + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.2, n_rows * 2.8))
        axes_flat = np.array(axes).flatten()

        for h in range(n_heads):
            ax = axes_flat[h]
            ax.imshow(attn[h].cpu().numpy(), cmap="Blues", vmin=0, vmax=1)
            ax.set_title(f"L{layer}H{h}", fontsize=8)
            ax.set_xticks(range(seq_len))
            ax.set_xticklabels(token_strs, rotation=90, fontsize=5)
            ax.set_yticks(range(seq_len))
            ax.set_yticklabels(token_strs, fontsize=5)
        for i in range(n_heads, len(axes_flat)):
            axes_flat[i].set_visible(False)

        fig.suptitle(f"Attention Patterns — Layer {layer}", fontsize=10)
        plt.tight_layout()
        plot_path = output_dir / f"attention_layer_{layer}.png"
        fig.savefig(plot_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        plot_paths.append(plot_path)

    return ExperimentResult(
        name="Attention Pattern",
        tool="attention_pattern",
        model_name=model.cfg.model_name,
        prompts=prompts,
        plot_paths=plot_paths,
        data={"layers_analysed": layers, "token_strs": token_strs, **layer_data},
        status="success",
    )
