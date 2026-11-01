"""What each pathway takes as input, for running one as a job.

The job store, the worker and the dashboard all need to know a pathway's
inputs without importing torch, so they live here rather than being read off
each module's `run()` signature. `run_from_params` is the one place that does
import the experiment stack, and only when a job actually runs.

Every pathway is given aligned lists: one positive/negative token per prompt,
and one contrast prompt per prompt where it takes them. `validate` checks
that before a job is queued, so a malformed request fails at the API rather
than minutes later in the worker.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple


class Pathway(NamedTuple):
    summary: str
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()
    # Optional inputs that must line up one-to-one with `prompts` when given.
    aligned_optional: tuple[str, ...] = ()


PATHWAYS: dict[str, Pathway] = {
    "circuit_discovery": Pathway(
        "what mechanism does the model use for this task? nominate heads, verify by patching, "
        "check the circuit is faithful",
        required=("prompts", "positive_tokens", "negative_tokens"),
        optional=("corrupted_prompts", "reference_prompts"),
        aligned_optional=("corrupted_prompts", "reference_prompts"),
    ),
    "attribute_audit": Pathway(
        "does the model use an attribute, and does removing its direction remove the behaviour?",
        required=("prompts", "counterfactual_prompts", "positive_tokens", "negative_tokens"),
        optional=("control_prompts", "control_texts"),
    ),
    "feature_explain": Pathway(
        "which SAE features drive this output, and what are they? (gpt2 only: uses a GPT-2 small SAE)",
        required=("prompts", "positive_tokens", "negative_tokens"),
    ),
}

# Inputs that have to line up with `prompts`, entry for entry.
_ALIGNED = ("positive_tokens", "negative_tokens", "counterfactual_prompts")

# feature_explain's SAE. Fixed for now: it's the one the known-answer test uses.
SAE_RELEASE, SAE_ID = "gpt2-small-res-jb", "blocks.8.hook_resid_pre"
SAE_MODELS = ("gpt2", "gpt2-small")


def names() -> list[str]:
    return list(PATHWAYS)


def validate(name: str, params: dict, model: str = "gpt2") -> list[str]:
    """Everything wrong with these inputs, or [] when the job can run."""
    pathway = PATHWAYS.get(name)
    if pathway is None:
        return [f"unknown pathway {name!r}; expected one of {names()}"]
    if not isinstance(params, dict):
        return ["params must be an object"]

    errors: list[str] = []
    allowed = set(pathway.required) | set(pathway.optional)
    for key in sorted(set(params) - allowed):
        errors.append(f"{name} does not take {key!r}")

    for key in pathway.required:
        value = params.get(key)
        if not _is_str_list(value) or not value:
            errors.append(f"{key} must be a non-empty list of strings")
    for key in pathway.optional:
        if key in params and params[key] is not None and not _is_str_list(params[key]):
            errors.append(f"{key} must be a list of strings")

    prompts = params.get("prompts")
    if _is_str_list(prompts) and prompts:
        for key in (*_ALIGNED, *pathway.aligned_optional):
            value = params.get(key)
            if _is_str_list(value) and value and len(value) != len(prompts):
                errors.append(f"{key} has {len(value)} entries but prompts has {len(prompts)}")

    if name == "feature_explain" and model not in SAE_MODELS:
        errors.append(f"feature_explain only runs on gpt2 (its SAE is {SAE_RELEASE}), not {model!r}")
    return errors


def run_from_params(name: str, model_name: str, params: dict, output_dir: Path):
    """Load the model (and SAE) and run the pathway. Imports torch."""
    problems = validate(name, params, model_name)
    if problems:
        raise ValueError("; ".join(problems))

    import importlib

    from transformer_lens import HookedTransformer

    module = importlib.import_module(f"seesaw.pathways.{name}")
    model = HookedTransformer.from_pretrained(model_name)
    kwargs = {k: v for k, v in params.items() if v}
    if name == "feature_explain":
        from seesaw.interp.feature_extraction.sae import load_sae

        sae = load_sae(SAE_RELEASE, SAE_ID)
        return module.run(model, sae, output_dir=output_dir, **kwargs)
    return module.run(model, output_dir=output_dir, **kwargs)


def _is_str_list(value) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)
