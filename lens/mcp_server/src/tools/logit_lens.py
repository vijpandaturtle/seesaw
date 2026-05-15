from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

matplotlib.use("Agg")   # non-interactive backend — saves to file

from ..app.schemas import ExperimentResult
from ..config import PLOTS_DIR


def run_logit_lens(
    model: HookedTransformer,
    prompts: list[str],
    pos: int = -1,
    top_k: int = 5,
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Project residual stream at each layer to vocab space.

    Shows how the model's prediction evolves layer by layer.
    Answers: "Where does the model 'decide' on the answer?"

    Args:
        model: Loaded HookedTransformer.
        prompts: List of input prompt strings.
        pos: Token position to inspect (-1 = last token).
        top_k: Number of top tokens to track per layer.
        output_dir: Directory to save the plot.

    Returns:
        ExperimentResult with a heatmap plot and layer-by-layer top-token data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens   = model.to_tokens(prompts)
    n_layers = model.cfg.n_layers

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    layer_data = []
    for layer in range(n_layers):
        resid = cache["resid_post", layer][:, pos, :]       # [batch, d_model]
        with torch.no_grad():
            logits = model.unembed(model.ln_final(resid))   # [batch, d_vocab]
            probs  = F.softmax(logits, dim=-1)
        top_probs, top_ids = probs[0].topk(top_k)
        layer_data.append({
            "layer":      layer,
            "top_tokens": [model.to_string(t.item()) for t in top_ids],
            "top_probs":  top_probs.tolist(),
        })

    # Collect unique tokens across all layers for the heatmap y-axis
    seen, all_tokens = set(), []
    for ld in layer_data:
        for t in ld["top_tokens"]:
            if t not in seen:
                seen.add(t)
                all_tokens.append(t)
    all_tokens = all_tokens[:15]

    prob_matrix = np.zeros((len(all_tokens), n_layers))
    for j, ld in enumerate(layer_data):
        for token, prob in zip(ld["top_tokens"], ld["top_probs"]):
            if token in all_tokens:
                prob_matrix[all_tokens.index(token), j] = prob

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(prob_matrix, aspect="auto", cmap="Blues",
                   vmin=0, vmax=max(prob_matrix.max(), 0.01))
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([str(l) for l in range(n_layers)], fontsize=8)
    ax.set_yticks(range(len(all_tokens)))
    ax.set_yticklabels([repr(t) for t in all_tokens], fontsize=9)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Token")
    ax.set_title(f"Logit Lens — {model.cfg.model_name}\n'{prompts[0][:70]}'")
    plt.colorbar(im, ax=ax, label="Probability")
    plt.tight_layout()

    plot_path = output_dir / "logit_lens.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return ExperimentResult(
        name="Logit Lens",
        tool="logit_lens",
        model_name=model.cfg.model_name,
        prompts=prompts,
        plot_paths=[plot_path],
        data={
            "layers":            layer_data,
            "position":          pos,
            "final_top_token":   layer_data[-1]["top_tokens"][0],
        },
        status="success",
    )
