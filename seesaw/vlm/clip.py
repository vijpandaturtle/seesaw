"""CLIP: which heads write the predicted label, and does removing them change the prediction?

CLIP has no vocabulary: it classifies by cosine similarity between the image
embedding and text embeddings of candidate labels. So here `prompts` are those
labels ("a photo of a dog", "a photo of a cat", ...), and every image is scored
against all of them.

- direct_effects (descriptive) — lever's exact decomposition of the image
  embedding into embed + per-head + per-MLP terms (Gandelsman et al. 2023),
  each projected onto the labels: how much each component *directly* pushes the
  predicted label above the average label. Direct only; a head that matters by
  feeding later layers shows up small.
- head_ablation (causal) — zero named heads (each alone, then all together) and
  measure the drop in the predicted label's probability and how often the
  prediction flips.

Head names follow seesaw's convention, "L<layer>H<head>", e.g. "L11H4".
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch

matplotlib.use("Agg")

from seesaw.interp.result import InterpResult

from .inputs import load_image

TOP_K = 12


def _component_name(name: str) -> str:
    """lever's "L3.H5" / "L3.mlp" -> seesaw's "L3H5" / "L3MLP"."""
    return name.replace(".H", "H").replace(".mlp", "MLP").replace(".attn_bias", "ATTN_BIAS")


def _parse_head(name: str) -> tuple[int, int]:
    m = re.fullmatch(r"L(\d+)H(\d+)", name.strip())
    if not m:
        raise ValueError(f"bad head name {name!r}; expected e.g. 'L11H4'")
    return int(m[1]), int(m[2])


def _scores(clip, text: torch.Tensor, pv: torch.Tensor) -> torch.Tensor:
    """Label probabilities [n_labels] for one preprocessed image."""
    from lever import label_probs

    return label_probs(clip, clip.encode_image(pv), text)[0].cpu()


@torch.no_grad()
def direct_effects(clip, prompts: list[str], images: list[str], output_dir: Path) -> InterpResult:
    from lever import decompose_image_embedding

    output_dir.mkdir(parents=True, exist_ok=True)
    text = clip.encode_text(prompts)
    grid = torch.zeros(clip.n_layers, clip.n_heads)
    totals: dict[str, float] = {}
    predictions = []
    for ref in images:
        pv = clip.pixels(load_image(ref))
        comps, names = decompose_image_embedding(clip, pv)       # [1, n, d], sums to the embedding
        emb = comps.sum(1)
        contrib = clip.logit_scale * (comps[0] @ text.T) / emb.norm()   # [n, labels]: sums to the label logits
        logits = contrib.sum(0)
        pred = int(logits.argmax())
        probs = logits.softmax(-1)
        predictions.append({"image": ref, "label": prompts[pred], "prob": round(probs[pred].item(), 4)})
        margin = (contrib[:, pred] - contrib.mean(1)).cpu()      # push toward the prediction over the average label
        for name, v in zip(names, margin.tolist()):
            key = _component_name(name)
            totals[key] = totals.get(key, 0.0) + v / len(images)
            if m := re.fullmatch(r"L(\d+)H(\d+)", key):
                grid[int(m[1]), int(m[2])] += v / len(images)

    ranked = sorted(((k, v) for k, v in totals.items() if k not in ("embed", "ln_bias")),
                    key=lambda kv: -abs(kv[1]))[:TOP_K]
    top_heads = [k for k, v in sorted(((k, v) for k, v in totals.items() if re.fullmatch(r"L\d+H\d+", k)),
                                      key=lambda kv: -kv[1])[:TOP_K]]

    fig, ax = plt.subplots(figsize=(max(5, 0.45 * clip.n_heads + 2), max(4, 0.35 * clip.n_layers + 1)))
    lim = grid.abs().max().item() or 1.0
    im = ax.imshow(grid.numpy(), cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title("CLIP Direct Effects — push toward the predicted label")
    fig.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout()
    plot_path = output_dir / "clip_direct_effects.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return InterpResult(
        plot_paths=[plot_path],
        data={
            "predictions":    predictions,
            "top_components": [{"component": k, "direct_effect": round(v, 4)} for k, v in ranked],
            "top_heads":      top_heads,
            "head_grid":      [[round(v, 4) for v in row] for row in grid.tolist()],
            "labels":         prompts,
        },
    )


@contextmanager
def _zero_heads(clip, heads: list[tuple[int, int]]):
    """Zero each head's slice of the attention output projection's input, at every position."""
    d_head = clip.layers[0].self_attn.out_proj.in_features // clip.n_heads
    by_layer: dict[int, list[int]] = {}
    for l, h in heads:
        by_layer.setdefault(l, []).append(h)

    def hook(hs):
        def pre(_m, args):
            z = args[0].clone()
            for h in hs:
                z[..., h * d_head:(h + 1) * d_head] = 0
            return (z, *args[1:])
        return pre

    handles = [clip.layers[l].self_attn.out_proj.register_forward_pre_hook(hook(hs)) for l, hs in by_layer.items()]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def head_ablation(clip, prompts: list[str], images: list[str], heads: list[str], output_dir: Path) -> InterpResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed = [_parse_head(h) for h in heads]
    for l, h in parsed:
        if l >= clip.n_layers or h >= clip.n_heads:
            raise ValueError(f"L{l}H{h} is out of range for {clip.n_layers} layers x {clip.n_heads} heads")
    text = clip.encode_text(prompts)
    conditions = [(h, [p]) for h, p in zip(heads, parsed)] + [("all together", parsed)]
    drop = {c: 0.0 for c, _ in conditions}
    flips = {c: 0 for c, _ in conditions}
    for ref in images:
        pv = clip.pixels(load_image(ref))
        base = _scores(clip, text, pv)
        pred = int(base.argmax())
        for cond, hs in conditions:
            with _zero_heads(clip, hs):
                after = _scores(clip, text, pv)
            drop[cond] += (base[pred] - after[pred]).item() / len(images)
            flips[cond] += int(after.argmax() != pred)

    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(conditions) + 2), 4))
    ax.bar(range(len(conditions)), [drop[c] for c, _ in conditions],
           color=["#4575b4"] * len(heads) + ["#d73027"])
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([c for c, _ in conditions], rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("drop in predicted label's probability")
    ax.set_title("CLIP Head Ablation (zero)")
    plt.tight_layout()
    plot_path = output_dir / "clip_head_ablation.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return InterpResult(
        plot_paths=[plot_path],
        data={
            "prob_drop":     {c: round(v, 4) for c, v in drop.items()},
            "flip_rate":     {c: round(flips[c] / len(images), 4) for c in flips},
            "heads":         heads,
            "n_images":      len(images),
            "labels":        prompts,
        },
    )
