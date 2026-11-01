"""Agent-facing wrapper over seesaw.interp.circuit_finding.circuit_evaluation."""

from pathlib import Path

from transformer_lens import HookedTransformer

from seesaw.interp.circuit_finding import circuit_evaluation as _circuit_evaluation

from ..config import PLOTS_DIR
from ..models.schemas import ExperimentResult
from ._wrap import wrap


def run_circuit_evaluation(
    model: HookedTransformer,
    prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    heads: list[str],
    reference_prompts: list[str] | None = None,
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Score a set of heads as a circuit: faithfulness and necessity, each against random head sets.

    Answers: "Is this set of heads the circuit, or only part of it?"
    """
    result = _circuit_evaluation.compute(model, prompts, positive_tokens, negative_tokens, heads,
                                         output_dir, reference_prompts=reference_prompts)
    return wrap(result, "Circuit Evaluation", "circuit_evaluation", model, prompts)
