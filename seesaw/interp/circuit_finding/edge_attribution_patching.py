"""Edge attribution patching: score every connection between components, not just the components.

Node-level tools say L8H6 matters. Edge-level ones say *how*: whether it
matters through L9H9's query or L10H0's, which is what makes a set of heads a
circuit. EAP (Syed et al. 2023) estimates every edge's patching effect from
gradients:

    score(u → v) = Σ_pos (clean_out_u - corrupt_out_u) · ∂metric/∂in_v

where out_u is what sender u writes to the residual stream and in_v is what
receiver v reads (a head's q, k or v input, an MLP's input, or the final
residual stream). On GPT-2 small that's ~70,000 edges from one backward pass.

`ig_steps > 0` gives EAP-IG (Hanna et al. 2024): gradients averaged along a
path from the corrupted to the clean input embeddings instead of taken at the
corrupted run alone. More faithful where the metric saturates, at the cost of
`ig_steps` backward passes.

Scores are normalised like activation patching's recovery: an edge scoring
0.3 would, by this estimate, restore 30% of the clean-vs-corrupt gap if it
alone were patched. It's an estimate; `path_patching` measures the top edges.

Answers: "Which connections between heads carry the behaviour?"
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from transformer_lens import HookedTransformer

matplotlib.use("Agg")

from .._helpers import tokens_to_ids
from ..metrics import logit_diff
from ..result import InterpResult
from ._edges import is_sender_hook, receivers, sender_output, senders, split_inputs


def compute(
    model: HookedTransformer,
    prompts: list[str],
    corrupted_prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    output_dir: Path,
    top_k: int = 30,
    ig_steps: int = 5,
) -> InterpResult:
    """Estimate the patching effect of every edge in the model.

    Args:
        model: Loaded HookedTransformer.
        prompts: Clean prompts.
        corrupted_prompts: Position-aligned corrupted prompts.
        positive_tokens: Correct answer token per prompt.
        negative_tokens: Distractor token per prompt.
        output_dir: Directory to save the plot.
        top_k: Edges to report, by absolute score.
        ig_steps: 0 for plain EAP; otherwise EAP-IG with this many steps.

    Returns:
        InterpResult with the top edges ("L5H5->L8H6.v", score), the total
        number of edges scored, and a circuit diagram of the top edges.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_tokens = model.to_tokens(prompts)
    corrupt_tokens = model.to_tokens(corrupted_prompts)
    if clean_tokens.shape != corrupt_tokens.shape:
        raise ValueError("clean and corrupted prompts must tokenize to the same shape")
    pos_ids = tokens_to_ids(model, positive_tokens)
    neg_ids = tokens_to_ids(model, negative_tokens)
    S, R = senders(model), receivers(model)

    with split_inputs(model):
        with torch.no_grad():
            clean_logits, clean_cache = model.run_with_cache(clean_tokens, names_filter=is_sender_hook)
            corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_tokens, names_filter=is_sender_hook)
        clean_ld = logit_diff(clean_logits, pos_ids, neg_ids).mean().item()
        corrupt_ld = logit_diff(corrupt_logits, pos_ids, neg_ids).mean().item()

        # [n_senders, batch, seq, d_model]: what each sender writes, clean minus corrupted.
        delta = torch.stack([
            sender_output(model, clean_cache, s) - sender_output(model, corrupt_cache, s) for s in S
        ])
        clean_embed = clean_cache["blocks.0.hook_resid_pre"]
        corrupt_embed = corrupt_cache["blocks.0.hook_resid_pre"]
        del clean_cache, corrupt_cache

        hooks_needed = sorted({r.hook for r in R})
        grads = {h: torch.zeros(()) for h in hooks_needed}
        alphas = [(k + 0.5) / ig_steps for k in range(ig_steps)] if ig_steps > 0 else [0.0]
        for alpha in alphas:
            saved: dict[str, torch.Tensor] = {}

            def root(act, hook, a=alpha):
                # The embedding, moved `a` of the way from corrupted to clean,
                # re-rooted so gradients exist even with frozen parameters.
                return (corrupt_embed + a * (clean_embed - corrupt_embed)).detach().requires_grad_(True)

            def keep(act, hook):
                saved[hook.name] = act
                return act

            with torch.enable_grad():
                with model.hooks(fwd_hooks=[("blocks.0.hook_resid_pre", root)]
                                 + [(h, keep) for h in hooks_needed]):
                    logits = model(corrupt_tokens)
                metric = logit_diff(logits, pos_ids, neg_ids).mean()
                g = torch.autograd.grad(metric, [saved[h] for h in hooks_needed])
            for h, gh in zip(hooks_needed, g):
                grads[h] = grads[h] + gh.detach() / len(alphas)

    # [n_receivers, batch, seq, d_model]: the gradient at each receiver's input.
    G = torch.stack([grads[r.hook][:, :, r.head, :] if r.head is not None else grads[r.hook] for r in R])
    del grads
    gap = clean_ld - corrupt_ld
    scale = gap if abs(gap) > 1e-6 else 1.0
    # The metric is a batch mean, so gradients already carry 1/batch: sum, don't average.
    scores = torch.einsum("nbsd,mbsd->nm", delta.float(), G.float()).cpu() / scale
    s_order = torch.tensor([s.order for s in S])[:, None]
    r_order = torch.tensor([r.order for r in R])[None, :]
    valid = s_order < r_order
    scores[~valid] = 0.0

    flat = scores.flatten()
    order = flat.abs().argsort(descending=True)[:top_k]
    top = [(f"{S[i // len(R)].name}->{R[i % len(R)].name}", round(flat[i].item(), 4)) for i in order.tolist()]

    plot_path = _plot(model, top, output_dir, clean_ld, corrupt_ld, ig_steps)
    return InterpResult(
        plot_paths=[plot_path],
        data={
            "clean_ld":        round(clean_ld, 4),
            "corrupt_ld":      round(corrupt_ld, 4),
            "method":          f"EAP-IG ({ig_steps} steps)" if ig_steps > 0 else "EAP",
            "n_edges_scored":  int(valid.sum()),
            "top_edges":       top,
            "positive_tokens": positive_tokens,
            "negative_tokens": negative_tokens,
            "corrupted_prompts": corrupted_prompts,
        },
    )


def _plot(model, top, output_dir, clean_ld, corrupt_ld, ig_steps) -> Path:
    """Circuit diagram: heads placed by (head, layer), MLPs at the right, embed below, logits above."""
    n_heads, n_layers = model.cfg.n_heads, model.cfg.n_layers

    def pos(name: str) -> tuple[float, float]:
        name = name.split(".")[0]
        if name == "embed":
            return (n_heads / 2, -1)
        if name == "logits":
            return (n_heads / 2, n_layers)
        if name.startswith("MLP"):
            return (n_heads + 0.5, int(name[3:]) + 0.4)
        layer, head = name.lstrip("L").split("H")
        return (int(head), int(layer))

    fig, ax = plt.subplots(figsize=(10, 9))
    wmax = max([abs(s) for _, s in top] + [1e-9])
    nodes = set()
    for edge, score in top:
        src, dst = edge.split("->")
        (x0, y0), (x1, y1) = pos(src), pos(dst)
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.5 + 3 * abs(score) / wmax,
                                    color="#d73027" if score > 0 else "#4575b4",
                                    alpha=0.35 + 0.6 * abs(score) / wmax, shrinkA=6, shrinkB=6))
        nodes |= {src, dst.split(".")[0]}
    for n in nodes:
        x, y = pos(n)
        ax.scatter([x], [y], s=120, color="#fdae61", edgecolor="black", zorder=3)
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.set_xlim(-1, n_heads + 2)
    ax.set_ylim(-1.7, n_layers + 0.7)
    ax.set_xlabel("Head (MLPs at right)")
    ax.set_ylabel("Layer")
    method = f"EAP-IG, {ig_steps} steps" if ig_steps > 0 else "EAP"
    ax.set_title(f"Edge Attribution Patching ({method}) — top {len(top)} edges\n"
                 f"red = restores the clean answer, blue = works against it  "
                 f"(clean LD {clean_ld:.2f}, corrupt {corrupt_ld:.2f})")
    plt.tight_layout()
    path = output_dir / "edge_attribution_patching.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
