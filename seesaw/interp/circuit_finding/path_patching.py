"""Path patching: patch one edge — a sender's output, only on its way into one receiver — and measure.

The verification half for edge attribution patching. For an edge u → v, the
corrupted run is repeated with v's input corrected by exactly u's
clean-minus-corrupted output; nothing else in the model sees the change. So
the measured recovery is that one connection's effect, the quantity EAP
estimates (Goldowsky-Dill et al. 2023; this is the single-edge form ACDC uses).

Each edge is patched alone, then all listed edges together: like heads,
edges in a circuit are often redundant, so the joint number can be much
larger than any single one.

Answers: "Does this specific connection carry the behaviour?"
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from transformer_lens import HookedTransformer

matplotlib.use("Agg")

from .._helpers import tokens_to_ids
from ..metrics import logit_diff, recovery
from ..result import InterpResult
from ._edges import is_sender_hook, parse_edge, sender_output, split_inputs


def compute(
    model: HookedTransformer,
    prompts: list[str],
    corrupted_prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    edges: list[str],
    output_dir: Path,
) -> InterpResult:
    """Patch each listed edge (and then all of them) from clean into the corrupted run.

    Args:
        model: Loaded HookedTransformer.
        prompts: Clean prompts.
        corrupted_prompts: Position-aligned corrupted prompts.
        positive_tokens: Correct answer token per prompt.
        negative_tokens: Distractor token per prompt.
        edges: Edges to test, e.g. ["L8H6->L9H9.q", "L9H9->logits"].
        output_dir: Directory to save the plot.

    Returns:
        InterpResult with per-edge and joint recovery (0 = none, 1 = full).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_tokens = model.to_tokens(prompts)
    corrupt_tokens = model.to_tokens(corrupted_prompts)
    pos_ids = tokens_to_ids(model, positive_tokens)
    neg_ids = tokens_to_ids(model, negative_tokens)
    parsed = [parse_edge(e, model) for e in edges]

    with split_inputs(model):
        with torch.no_grad():
            clean_logits, clean_cache = model.run_with_cache(clean_tokens, names_filter=is_sender_hook)
            corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_tokens, names_filter=is_sender_hook)
        clean_ld = logit_diff(clean_logits, pos_ids, neg_ids).mean().item()
        corrupt_ld = logit_diff(corrupt_logits, pos_ids, neg_ids).mean().item()
        delta = {
            s.name: sender_output(model, clean_cache, s) - sender_output(model, corrupt_cache, s)
            for s, _ in parsed
        }
        del clean_cache, corrupt_cache

        def patched_ld(pairs) -> float:
            by_hook: dict[str, list[tuple[int | None, torch.Tensor]]] = {}
            for s, r in pairs:
                by_hook.setdefault(r.hook, []).append((r.head, delta[s.name]))

            def make_hook(adds):
                def hook_fn(value, hook):
                    value = value.clone()
                    for head, d in adds:
                        if head is None:
                            value = value + d
                        else:
                            value[:, :, head, :] = value[:, :, head, :] + d
                    return value
                return hook_fn

            with torch.no_grad(), model.hooks(fwd_hooks=[(h, make_hook(a)) for h, a in by_hook.items()]):
                return logit_diff(model(corrupt_tokens), pos_ids, neg_ids).mean().item()

        per_edge = {e: round(recovery(patched_ld([p]), corrupt_ld, clean_ld), 4) for e, p in zip(edges, parsed)}
        joint = round(recovery(patched_ld(parsed), corrupt_ld, clean_ld), 4) if parsed else 0.0

    # ── Plot ──────────────────────────────────────────────────────────────────
    labels = list(per_edge) + ["all together"]
    values = list(per_edge.values()) + [joint]
    fig, ax = plt.subplots(figsize=(max(7, 0.45 * len(labels) + 2), 4.5))
    ax.bar(range(len(labels)), values, color=["#4575b4"] * len(per_edge) + ["#d73027"])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Fraction of logit diff recovered")
    ax.set_title("Path Patching — one edge at a time, clean → corrupted")
    plt.tight_layout()
    plot_path = output_dir / "path_patching.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return InterpResult(
        plot_paths=[plot_path],
        data={
            "clean_ld":       round(clean_ld, 4),
            "corrupt_ld":     round(corrupt_ld, 4),
            "per_edge":       per_edge,
            "joint_recovery": joint,
            "edges":          edges,
            "corrupted_prompts": corrupted_prompts,
        },
    )
