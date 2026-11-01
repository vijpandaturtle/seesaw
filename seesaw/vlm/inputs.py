"""Turning plan arguments into model inputs: image references, chat-template inputs, answer tokens.

A research plan is text, so images arrive as references. Each is one of:

    /abs/path.png  or  photo.jpg      a file; relative paths resolve against SEESAW_IMAGE_DIR
                                      (default: the working directory)
    https://…/x.jpg                   downloaded once and cached
    text:<words>                      the words on a white canvas
    solid:<colour>                    a plain image of one colour (e.g. solid:red)
    <any of the above>+text:<words>   that image with the words stamped on it: a typographic attack

The synthetic forms let a planner build controlled stimuli and counterfactuals
(a picture vs the same picture with "iPod" written on it) without a dataset.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import torch
from PIL import Image

CANVAS = (336, 336)   # LLaVA-1.5's input size; other models resize anyway
STAMP_FRAC = 0.12     # label height as a fraction of the image, as in lever's typographic experiments
_CACHE = Path(os.environ.get("SEESAW_IMAGE_CACHE", Path.home() / ".cache/seesaw/images"))


def load_image(ref: str) -> Image.Image:
    """Resolve one image reference (see the module docstring) to an RGB image."""
    ref = ref.strip()
    base, sep, stamp = ref.rpartition("+text:")
    if sep and base:
        from lever.images import stamp_text

        return stamp_text(load_image(base), stamp, STAMP_FRAC)
    if ref.startswith("text:"):
        from lever.images import stamp_text

        return stamp_text(Image.new("RGB", CANVAS, "white"), ref[5:], STAMP_FRAC)
    if ref.startswith("solid:"):
        return Image.new("RGB", CANVAS, ref[6:])
    if ref.startswith(("http://", "https://")):
        return Image.open(_download(ref)).convert("RGB")
    path = Path(ref).expanduser()
    if not path.is_absolute():
        path = Path(os.environ.get("SEESAW_IMAGE_DIR", ".")) / path
    if not path.exists():
        raise FileNotFoundError(f"image {ref!r} not found (looked at {path}); see seesaw/vlm/inputs.py "
                                "for the reference forms, e.g. 'text:stop' or 'solid:red'")
    return Image.open(path).convert("RGB")


def _download(url: str) -> Path:
    _CACHE.mkdir(parents=True, exist_ok=True)
    path = _CACHE / hashlib.sha256(url.encode()).hexdigest()[:24]
    if not path.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "seesaw"})
        with urllib.request.urlopen(req, timeout=30) as r:
            path.write_bytes(r.read())
    return path


def broadcast(values: list[str] | str, n: int, what: str) -> list[str]:
    """One value per prompt: a single value is applied to every prompt."""
    values = [values] if isinstance(values, str) else list(values)
    if len(values) == 1:
        return values * n
    if len(values) != n:
        raise ValueError(f"need one {what} per prompt (or one for all): got {len(values)} for {n} prompts")
    return values


def prepare(vlm, image: Image.Image, prompt: str) -> dict[str, torch.Tensor]:
    """Model inputs exactly as the model's chat template builds them, ending at the
    assistant's turn — so the last position's logits are the answer's first token."""
    return vlm.prepare(image, prompt)


def token_id(vlm, token: str) -> int:
    """The first token of an answer string. Chat models start their answer with no
    leading space ("Yes", not " Yes"), unlike GPT-2 prompt continuations."""
    tok = getattr(vlm.processor, "tokenizer", None)
    if tok is None:          # tiny test models have no tokenizer; their "tokens" are ids
        return int(token)
    ids = tok.encode(token, add_special_tokens=False)
    if not ids:
        raise ValueError(f"{token!r} tokenizes to nothing")
    return ids[0]


def answer_logits(vlm, inputs: dict) -> torch.Tensor:
    """Next-token logits at the last position, as float32 on CPU."""
    return vlm.model(**inputs).logits[0, -1].float().cpu()


def same_size(image: Image.Image, like: Image.Image) -> Image.Image:
    """Resize a counterfactual image to its clean partner's size, so both become the
    same number of image tokens and positions line up for patching."""
    return image if image.size == like.size else image.resize(like.size)
