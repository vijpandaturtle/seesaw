"""Causal tools for VLMs: does the answer need the image, when is it read, and where does it flow?

All three change the forward pass with `lever.patch_output` and measure
logit(positive) - logit(negative) at the answer position. Everything downstream
of the edit recomputes, so these are total effects, not direct ones.

- vision_ablation — zero or mean-ablate the projector's output (everything the
  vision tower hands the LM). What's left is the language prior: the answer the
  model gives from the question alone.
- image_token_ablation — after each LM layer, overwrite the image-token
  positions with their mean, wiping their content from that layer on. The
  effect that survives is what text positions had already read out of the
  image by then: where image→text transfer finishes.
- activation_patching — run a corrupted image (the same scene with one thing
  changed), patch one LM layer's residual from the clean run at image, text or
  answer positions, and measure how much of the clean answer comes back.

Caveat (lever): on Qwen3-VL, DeepStack injects vision features into early LM
layers directly, so ablating the projector does not remove all visual input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import torch

matplotlib.use("Agg")

from seesaw.interp.result import InterpResult

from .inputs import answer_logits, broadcast, load_image, prepare, same_size, token_id

# A clean-vs-baseline gap smaller than this (logits) is too small to normalise against.
MIN_GAP = 0.1
# image_token_ablation: the layer from which wiping image tokens keeps this much
# of the image's effect is where transfer is reported as complete.
TRANSFER_DONE = 0.9
GROUPS = ("image", "text", "answer")


def _setup(vlm, prompts, images, positive_tokens, negative_tokens):
    n = len(prompts)
    images = broadcast(images, n, "image")
    pos = [token_id(vlm, t) for t in broadcast(positive_tokens, n, "positive token")]
    neg = [token_id(vlm, t) for t in broadcast(negative_tokens, n, "negative token")]
    return images, pos, neg


def _ld(vlm, inputs, p, q) -> float:
    logits = answer_logits(vlm, inputs)
    return (logits[p] - logits[q]).item()


def _ablate(kind: str):
    """Projector-output edit. Mean over every image token (LLaVA's output is
    [batch, tokens, d]; Qwen's merger flattens to [tokens, d])."""
    if kind == "zero":
        return torch.zeros_like
    return lambda x: x.reshape(-1, x.shape[-1]).mean(0).expand_as(x).clone()


def _deepstack(vlm) -> bool:
    return getattr(vlm.model.config, "model_type", "") == "qwen3_vl"


def _ylabel(positive_tokens, negative_tokens) -> str:
    first = lambda t: t if isinstance(t, str) else t[0]   # noqa: E731
    return f"logit({first(positive_tokens)!r}) - logit({first(negative_tokens)!r})"


@torch.no_grad()
def vision_ablation(
    vlm,
    prompts: list[str],
    images: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    output_dir: Path,
    ablation_type: Literal["zero", "mean"] = "mean",
) -> InterpResult:
    """Remove the vision tower's output and measure how much of the answer goes with it."""
    from lever import patch_output

    output_dir.mkdir(parents=True, exist_ok=True)
    images, pos, neg = _setup(vlm, prompts, images, positive_tokens, negative_tokens)
    clean, ablated = [], []
    for prompt, ref, p, q in zip(prompts, images, pos, neg):
        inputs = prepare(vlm, load_image(ref), prompt)
        clean.append(_ld(vlm, inputs, p, q))
        with patch_output(vlm.projector, _ablate(ablation_type)):
            ablated.append(_ld(vlm, inputs, p, q))

    mc, ma = sum(clean) / len(clean), sum(ablated) / len(ablated)
    removed = (mc - ma) / mc if abs(mc) >= MIN_GAP else None

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(prompts) + 2), 4))
    x = torch.arange(len(prompts))
    ax.bar(x - 0.2, clean, width=0.4, label="with image", color="#d73027")
    ax.bar(x + 0.2, ablated, width=0.4, label=f"vision {ablation_type}-ablated", color="#4575b4")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x.tolist())
    ax.set_xticklabels([f"{r[-20:]} | {p[-20:]}" for r, p in zip(images, prompts)],
                       rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(_ylabel(positive_tokens, negative_tokens))
    ax.set_title("Vision Ablation" + ("" if removed is None else f" — {removed:.0%} of the answer removed"))
    ax.legend(fontsize=8)
    plt.tight_layout()
    plot_path = output_dir / "vision_ablation.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    data = {
        "clean_logit_diff":     round(mc, 4),
        "ablated_logit_diff":   round(ma, 4),
        "fraction_removed":     None if removed is None else round(removed, 4),
        "per_prompt_clean":     [round(v, 4) for v in clean],
        "per_prompt_ablated":   [round(v, 4) for v in ablated],
        "ablation_type":        ablation_type,
        "images":               images,
    }
    if removed is None:
        data["note"] = f"clean logit diff {mc:+.2f} is under {MIN_GAP}: no answer preference to remove"
    if _deepstack(vlm):
        data["caveat"] = ("Qwen3-VL's DeepStack feeds vision features into early LM layers directly; "
                          "ablating the projector does not remove all visual input")
    return InterpResult(plot_paths=[plot_path], data=data)


@torch.no_grad()
def image_token_ablation(
    vlm,
    prompts: list[str],
    images: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    output_dir: Path,
) -> InterpResult:
    """Wipe the image tokens after each LM layer: by which layer has the answer read what it needs?"""
    from lever import patch_output

    output_dir.mkdir(parents=True, exist_ok=True)
    images, pos, neg = _setup(vlm, prompts, images, positive_tokens, negative_tokens)
    L = vlm.n_layers
    clean, no_image, by_layer = [], [], torch.zeros(len(prompts), L)
    for k, (prompt, ref, p, q) in enumerate(zip(prompts, images, pos, neg)):
        inputs = prepare(vlm, load_image(ref), prompt)
        mask = vlm.image_mask(inputs["input_ids"])
        if not mask.any():
            raise ValueError("no image tokens in the input; is this a vision-language model?")
        clean.append(_ld(vlm, inputs, p, q))
        with patch_output(vlm.projector, _ablate("mean")):
            no_image.append(_ld(vlm, inputs, p, q))

        def wipe(x, m=mask):
            x = x.clone()
            x[m] = x[m].mean(0, keepdim=True)
            return x

        for i in range(L):
            with patch_output(vlm.lm_layers[i], wipe):
                by_layer[k, i] = _ld(vlm, inputs, p, q)

    mc, mn = sum(clean) / len(clean), sum(no_image) / len(no_image)
    gap = mc - mn
    lds = by_layer.mean(0)
    kept = ((lds - mn) / gap).tolist() if abs(gap) >= MIN_GAP else None
    done = None if kept is None else next((i for i, v in enumerate(kept) if v >= TRANSFER_DONE), None)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(L), lds.numpy(), marker="o", color="#4575b4", label="image tokens wiped after layer")
    ax.axhline(mc, color="#d73027", linestyle="--", label=f"with image {mc:+.2f}")
    ax.axhline(mn, color="grey", linestyle=":", label=f"no image {mn:+.2f}")
    if done is not None:
        ax.axvline(done, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("LM layer")
    ax.set_ylabel(_ylabel(positive_tokens, negative_tokens))
    ax.set_title("Image-Token Ablation by Layer")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plot_path = output_dir / "image_token_ablation.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    data = {
        "clean_logit_diff":    round(mc, 4),
        "no_image_logit_diff": round(mn, 4),
        "layer_logit_diffs":   [round(v, 4) for v in lds.tolist()],
        "fraction_kept":       None if kept is None else [round(v, 4) for v in kept],
        "transfer_done_layer": done,
        "images":              images,
    }
    if kept is None:
        data["note"] = (f"the image moves the answer by only {gap:+.2f} logits, under {MIN_GAP}; "
                        "there's no image effect to localise")
    if _deepstack(vlm):
        data["caveat"] = ("Qwen3-VL's DeepStack adds vision features to early LM layers' residuals; "
                          "wiping image tokens before those layers is partly undone")
    return InterpResult(plot_paths=[plot_path], data=data)


@torch.no_grad()
def activation_patching(
    vlm,
    prompts: list[str],
    images: list[str],
    corrupted_images: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    output_dir: Path,
) -> InterpResult:
    """Patch clean LM residuals into a corrupted-image run, per layer and position group."""
    from lever import patch_output, record

    output_dir.mkdir(parents=True, exist_ok=True)
    images, pos, neg = _setup(vlm, prompts, images, positive_tokens, negative_tokens)
    corrupted_images = broadcast(corrupted_images, len(prompts), "corrupted image")
    L = vlm.n_layers
    points = {f"lm.{i}": layer for i, layer in enumerate(vlm.lm_layers)}
    clean, corrupt = [], []
    patched = torch.zeros(len(prompts), len(GROUPS), L)
    for k, (prompt, ref, cref, p, q) in enumerate(zip(prompts, images, corrupted_images, pos, neg)):
        img = load_image(ref)
        c_in = prepare(vlm, img, prompt)
        x_in = prepare(vlm, same_size(load_image(cref), img), prompt)
        if c_in["input_ids"].shape != x_in["input_ids"].shape:
            raise ValueError(f"clean and corrupted inputs differ in length ({c_in['input_ids'].shape[-1]} vs "
                             f"{x_in['input_ids'].shape[-1]}); positions can't be aligned for patching")
        img_mask = vlm.image_mask(c_in["input_ids"])
        answer = torch.zeros_like(img_mask)
        answer[:, -1] = True
        masks = {"image": img_mask, "text": ~img_mask & ~answer, "answer": answer}

        with record(points) as cache:
            clean.append(_ld(vlm, c_in, p, q))
        corrupt.append(_ld(vlm, x_in, p, q))
        for g, group in enumerate(GROUPS):
            m = masks[group]
            for i in range(L):
                def put(x, src=cache[f"lm.{i}"], m=m):
                    x = x.clone()
                    x[m] = src[m].to(x.dtype)
                    return x

                with patch_output(vlm.lm_layers[i], put):
                    patched[k, g, i] = _ld(vlm, x_in, p, q)

    mc, mx = sum(clean) / len(clean), sum(corrupt) / len(corrupt)
    gap = mc - mx
    if abs(gap) < MIN_GAP:
        raise ValueError(f"the corrupted images move the logit diff by only {gap:+.2f}, under {MIN_GAP}: "
                         "nothing to recover. Choose a corruption that changes the answer.")
    recovery = (patched.mean(0) - mx) / gap   # [groups, layers]

    fig, ax = plt.subplots(figsize=(max(8, 0.35 * L + 2), 2.8))
    im = ax.imshow(recovery.numpy(), cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(len(GROUPS)))
    ax.set_yticklabels([f"{g} positions" for g in GROUPS])
    ax.set_xlabel("LM layer")
    ax.set_title("Activation Patching — fraction of clean answer recovered (clean → corrupted image)")
    fig.colorbar(im, ax=ax, fraction=0.02)
    plt.tight_layout()
    plot_path = output_dir / "vlm_activation_patching.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    best = {g: (int(recovery[i].argmax()), round(recovery[i].max().item(), 4)) for i, g in enumerate(GROUPS)}
    return InterpResult(
        plot_paths=[plot_path],
        data={
            "clean_logit_diff":     round(mc, 4),
            "corrupted_logit_diff": round(mx, 4),
            "recovery":             {g: [round(v, 4) for v in recovery[i].tolist()] for i, g in enumerate(GROUPS)},
            "peak_layer_by_group":  {g: {"layer": l, "recovery": r} for g, (l, r) in best.items()},
            "images":               images,
            "corrupted_images":     corrupted_images,
        },
    )
