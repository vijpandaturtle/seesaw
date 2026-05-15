from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from transformer_lens import HookedTransformer

matplotlib.use("Agg")

from ..app.helpers import tokens_to_ids
from ..app.schemas import ExperimentResult
from ..config import PLOTS_DIR


def run_direct_logit_attribution(
    model: HookedTransformer,
    prompts: list[str],
    io_tokens: list[str],
    subject_tokens: list[str],
    pos: int = -1,
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Decompose the final logit difference into per-head and per-MLP contributions.

    Uses cache['z', layer] + W_O (always available, no extra flags needed).
    This avoids the use_attn_result flag which conflicts with
    refactor_factored_attn_matrices.

    Answers: "Which components *write* the IO prediction into the residual stream?"

    Args:
        model: Loaded HookedTransformer.
        prompts: List of input prompt strings.
        io_tokens: Correct (indirect object) token strings, e.g. [" Mary", " Tom"].
        subject_tokens: Incorrect (subject) token strings, e.g. [" John", " Sarah"].
        pos: Token position to read logits from (-1 = last token).
        output_dir: Directory to save the plot.

    Returns:
        ExperimentResult with head and MLP attribution heatmaps.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens   = model.to_tokens(prompts)
    n_layers = model.cfg.n_layers
    n_heads  = model.cfg.n_heads
    W_U      = model.W_U   # [d_model, d_vocab]

    io_ids = tokens_to_ids(model, io_tokens)
    s_ids  = tokens_to_ids(model, subject_tokens)

    # Build IO - Subject direction in vocab space, then detach to avoid requires_grad
    logit_diff_dir = torch.zeros(model.cfg.d_model)
    for io_id, s_id in zip(io_ids, s_ids):
        logit_diff_dir += W_U[:, io_id] - W_U[:, s_id]
    logit_diff_dir = logit_diff_dir.detach() / len(io_ids)   # detach: W_U is a Parameter

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    head_attrs = torch.zeros(n_layers, n_heads)
    mlp_attrs  = torch.zeros(n_layers)

    for layer in range(n_layers):
        # cache["z", layer]: [batch, seq, n_heads, d_head]  — always cached
        # model.W_O[layer]:  [n_heads, d_head, d_model]
        z        = cache["z", layer]
        W_O      = model.W_O[layer]
        head_out = torch.einsum("bshd,hdm->bshm", z, W_O)   # [B, S, H, d_model]

        for h in range(n_heads):
            h_vec = head_out[:, pos, h, :].mean(0)           # [d_model], mean over batch
            head_attrs[layer, h] = h_vec @ logit_diff_dir

        mlp_out = cache["mlp_out", layer][:, pos, :].mean(0)
        mlp_attrs[layer] = mlp_out @ logit_diff_dir

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    vmax = head_attrs.abs().max().item()
    im   = axes[0].imshow(head_attrs.detach().numpy(), cmap="RdBu", vmin=-vmax, vmax=vmax)
    axes[0].set_xlabel("Head")
    axes[0].set_ylabel("Layer")
    axes[0].set_title("Head Attribution (IO - Subject logit diff)")
    axes[0].set_xticks(range(n_heads))
    axes[0].set_yticks(range(n_layers))
    plt.colorbar(im, ax=axes[0])

    colors = ["#d73027" if v > 0 else "#4575b4" for v in mlp_attrs.tolist()]
    axes[1].barh(range(n_layers), mlp_attrs.detach().numpy(), color=colors)
    axes[1].set_xlabel("Logit diff contribution")
    axes[1].set_ylabel("Layer")
    axes[1].set_title("MLP Attribution per Layer")
    axes[1].set_yticks(range(n_layers))
    axes[1].axvline(0, color="black", linewidth=0.8)

    plt.suptitle(f"Direct Logit Attribution — {model.cfg.model_name}", fontsize=12)
    plt.tight_layout()
    plot_path = output_dir / "direct_logit_attribution.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    flat      = head_attrs.flatten()
    top_idx   = flat.abs().topk(5).indices
    top_heads = [(int(i // n_heads), int(i % n_heads), flat[i].item()) for i in top_idx]

    return ExperimentResult(
        name="Direct Logit Attribution",
        tool="direct_logit_attribution",
        model_name=model.cfg.model_name,
        prompts=prompts,
        plot_paths=[plot_path],
        data={
            "head_attrs":     head_attrs.tolist(),
            "mlp_attrs":      mlp_attrs.tolist(),
            "top_heads":      [(f"L{l}H{h}", round(v, 4)) for l, h, v in top_heads],
            "io_tokens":      io_tokens,
            "subject_tokens": subject_tokens,
        },
        status="success",
    )
