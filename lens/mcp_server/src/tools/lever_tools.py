"""Agent-facing wrappers over seesaw.vlm: Lens's tools when SEESAW_INTERP_BACKEND=lever.

Same contract as the TransformerLens wrappers (model and prompts supplied by
the node, everything else from tool_kwargs, an ExperimentResult back), so
dispatch, follow-ups and Quill need nothing backend-specific. lever is imported
only inside the calls, so the default backend runs without the [lever] extra.

`model` is a lever handle: a `lever.VLM` for the VLM tools, a `lever.CLIP` for
the clip_ ones. Each tool checks it got the kind it needs.
"""

from pathlib import Path
from typing import Literal

from seesaw import vlm as _vlm
from seesaw.interp.result import InterpResult
from seesaw.vlm import clip as _clip
from seesaw.vlm import interventions as _iv
from seesaw.vlm import logit_lens as _lens

from ..config import PLOTS_DIR
from ..models.schemas import ExperimentResult


def _wrap(result: InterpResult, name: str, tool: str, model, prompts: list[str]) -> ExperimentResult:
    return ExperimentResult(
        name=name,
        tool=tool,
        model_name=_vlm.model_label(model),
        prompts=prompts,
        plot_paths=result.plot_paths,
        data=result.data,
        status="success",
    )


def _need(model, clip: bool, tool: str) -> None:
    if _vlm.is_clip(model) != clip:
        want = "a CLIP checkpoint" if clip else "a vision-language model (LLaVA / Qwen-VL), not CLIP"
        raise TypeError(f"{tool} needs {want}")


def run_vlm_logit_lens(
    model,
    prompts: list[str],
    images: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Decode the LM residual after each layer at the answer position and at image tokens.

    Answers: "At which layer does the answer appear, and do image tokens decode to words?"
    """
    _need(model, False, "logit_lens")
    r = _lens.compute(model, prompts, images, positive_tokens, negative_tokens, output_dir)
    return _wrap(r, "Logit Lens (VLM)", "logit_lens", model, prompts)


def run_vision_ablation(
    model,
    prompts: list[str],
    images: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    ablation_type: Literal["zero", "mean"] = "mean",
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Ablate the vision tower's output: how much of the answer needs the image?"""
    _need(model, False, "vision_ablation")
    r = _iv.vision_ablation(model, prompts, images, positive_tokens, negative_tokens, output_dir,
                            ablation_type=ablation_type)
    return _wrap(r, f"Vision Ablation ({ablation_type})", "vision_ablation", model, prompts)


def run_image_token_ablation(
    model,
    prompts: list[str],
    images: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Wipe image tokens after each LM layer: by which layer is the image read?"""
    _need(model, False, "image_token_ablation")
    r = _iv.image_token_ablation(model, prompts, images, positive_tokens, negative_tokens, output_dir)
    return _wrap(r, "Image-Token Ablation by Layer", "image_token_ablation", model, prompts)


def run_vlm_activation_patching(
    model,
    prompts: list[str],
    images: list[str],
    corrupted_images: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Patch clean LM residuals into the corrupted-image run, per layer and position group."""
    _need(model, False, "activation_patching")
    r = _iv.activation_patching(model, prompts, images, corrupted_images, positive_tokens,
                                negative_tokens, output_dir)
    return _wrap(r, "Activation Patching (VLM)", "activation_patching", model, prompts)


def run_clip_direct_effects(
    model,
    prompts: list[str],
    images: list[str],
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Per-head and per-MLP direct effects on CLIP's predicted label; prompts are the candidate labels."""
    _need(model, True, "clip_direct_effects")
    r = _clip.direct_effects(model, prompts, images, output_dir)
    return _wrap(r, "CLIP Direct Effects", "clip_direct_effects", model, prompts)


def run_clip_head_ablation(
    model,
    prompts: list[str],
    images: list[str],
    heads: list[str],
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Zero named CLIP heads and measure the predicted label's probability drop and flip rate."""
    _need(model, True, "clip_head_ablation")
    r = _clip.head_ablation(model, prompts, images, heads, output_dir)
    return _wrap(r, "CLIP Head Ablation", "clip_head_ablation", model, prompts)


# Keys must match shared/lens_tools.py::LEVER_TOOLS (tests/test_lens_tools.py checks).
LEVER_REGISTRY: dict = {
    "logit_lens":           run_vlm_logit_lens,
    "clip_direct_effects":  run_clip_direct_effects,
    "vision_ablation":      run_vision_ablation,
    "image_token_ablation": run_image_token_ablation,
    "activation_patching":  run_vlm_activation_patching,
    "clip_head_ablation":   run_clip_head_ablation,
}
