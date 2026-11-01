"""Interpretability for vision-language models, built on lever — Lens's second backend.

`seesaw.interp` is written against TransformerLens, which doesn't wrap
vision-language architectures. This package runs the same kind of experiment
on LLaVA, Qwen-VL and CLIP through lever (github.com/vijpandaturtle/lever-interp),
which hooks the unmodified Hugging Face model, so what gets measured is the
model's own forward pass (its image preprocessing, token merging and RoPE).

Like `seesaw.interp`, nothing here knows about agents: every function takes a
loaded model plus plain arguments and returns an `InterpResult`.

    vlm/
    ├── inputs.py         — image references, chat-template inputs, answer tokens
    ├── logit_lens.py     — descriptive: where the answer forms, what image tokens decode to
    ├── interventions.py  — causal: vision ablation, image-token ablation by layer,
    │                       activation patching between a clean and a corrupted image
    └── clip.py           — CLIP: per-head direct effects on the predicted label, head ablation

Selected with SEESAW_INTERP_BACKEND=lever; needs the [lever] extra.
"""

from __future__ import annotations


def load(model_name: str):
    """Load a checkpoint through lever: CLIP as a `lever.CLIP`, anything else as a `lever.VLM`.

    Chosen by the checkpoint's own config rather than its name, so fine-tunes of
    a known architecture load without a lookup table.
    """
    import lever
    from transformers import AutoConfig

    if AutoConfig.from_pretrained(model_name).model_type == "clip":
        return lever.load_clip(model_name)
    return lever.load_vlm(model_name)


def is_clip(model) -> bool:
    import lever

    return isinstance(model, lever.CLIP)


def model_label(model) -> str:
    """The checkpoint id, for results: lever handles have no `cfg.model_name`."""
    return getattr(model.model.config, "_name_or_path", "") or type(model.model).__name__
