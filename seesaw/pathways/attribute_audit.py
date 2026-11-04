"""Attribute audit: the concept-direction pathway under the name older jobs were saved with.

Version 1 fitted its direction on the same prompt pairs it then tested it on,
so a difference of means could remove its own prompts' gap by construction.
Version 2 is `concept_direction`, which fits on half the pairs and tests on
the other half; this module keeps the name so earlier jobs can be rerun.
The verdicts are concept_direction's.
"""

from __future__ import annotations

from pathlib import Path

from transformer_lens import HookedTransformer

from . import concept_direction
from .result import PathwayResult


def run(
    model: HookedTransformer,
    prompts: list[str],
    counterfactual_prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    output_dir: Path,
    control_prompts: list[str] | None = None,
    control_texts: list[str] | None = None,
) -> PathwayResult:
    """Run concept_direction, recorded as attribute_audit."""
    return concept_direction.run(
        model, prompts, counterfactual_prompts, positive_tokens, negative_tokens, output_dir,
        control_prompts=control_prompts, control_texts=control_texts, pathway="attribute_audit")
