"""Logit lens on a VLM's language model: where the answer forms, and what image tokens decode to.

Two readouts from one cached forward pass per (image, prompt):

- At the last position, logit(positive) - logit(negative) after each LM layer:
  the layer at which the model "has" the answer.
- At image-token positions, the most common top-1 token after each layer. Image
  tokens have no next-token target, so this is the only direct way to ask
  whether a patch's representation looks like the word "dog" yet.

Descriptive only: the lens is biased toward late layers (early states aren't in
the unembedding's basis), and decoding a state isn't evidence the model uses it.

Answers: "At which layer does the answer appear, and do image tokens carry words?"
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch

matplotlib.use("Agg")

from seesaw.interp.result import InterpResult

from .inputs import broadcast, load_image, prepare, token_id

# Image-token readouts are reported every IMAGE_LAYER_STRIDE layers (plus the
# last), to keep the result short enough to read.
IMAGE_LAYER_STRIDE = 4


@torch.no_grad()
def compute(
    vlm,
    prompts: list[str],
    images: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    output_dir: Path,
) -> InterpResult:
    from lever import logit_lens, record

    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(prompts)
    images = broadcast(images, n, "image")
    pos = [token_id(vlm, t) for t in broadcast(positive_tokens, n, "positive token")]
    neg = [token_id(vlm, t) for t in broadcast(negative_tokens, n, "negative token")]
    points = {f"lm.{i}": layer for i, layer in enumerate(vlm.lm_layers)}

    layer_lds = torch.zeros(n, vlm.n_layers)
    final_lds, top_by_layer, image_top = [], [], {}
    for k, (prompt, ref) in enumerate(zip(prompts, images)):
        inputs = prepare(vlm, load_image(ref), prompt)
        with record(points, to_cpu=True) as cache:
            logits = vlm.model(**inputs).logits[0, -1].float()
        final_lds.append((logits[pos[k]] - logits[neg[k]]).item())
        img_mask = vlm.image_mask(inputs["input_ids"])[0].cpu()
        for i in range(vlm.n_layers):
            lp = logit_lens(vlm, cache[f"lm.{i}"][0, -1]).cpu()
            layer_lds[k, i] = lp[pos[k]] - lp[neg[k]]
            if k == 0:
                top_by_layer.append(vlm.decode([lp.argmax()])[0])
                if (i % IMAGE_LAYER_STRIDE == 0 or i == vlm.n_layers - 1) and img_mask.any():
                    ids = logit_lens(vlm, cache[f"lm.{i}"][0, img_mask]).argmax(-1).tolist()
                    image_top[i] = [(t, c) for t, c in Counter(vlm.decode(ids)).most_common(3)]

    mean_lds = layer_lds.mean(0)
    final = sum(final_lds) / n
    half = next((i for i, v in enumerate(mean_lds.tolist()) if final and v / final >= 0.5), None)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(vlm.n_layers), mean_lds.numpy(), marker="o", color="#4575b4")
    ax.axhline(final, color="#d73027", linestyle="--", label=f"model output {final:+.2f}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("LM layer")
    ax.set_ylabel(f"logit({positive_tokens[0]!r}) - logit({negative_tokens[0]!r})")
    ax.set_title("Logit Lens — answer position, by layer")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plot_path = output_dir / "vlm_logit_lens.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return InterpResult(
        plot_paths=[plot_path],
        data={
            "final_logit_diff":       round(final, 4),
            "layer_logit_diffs":      [round(v, 4) for v in mean_lds.tolist()],
            "first_layer_half_final": half,
            "top_token_by_layer":     top_by_layer,
            "image_token_top_tokens": image_top,
            "images":                 images,
            "positive_tokens":        positive_tokens,
            "negative_tokens":        negative_tokens,
        },
    )
