"""Task-agnostic graders for Scout, Lens, and Quill, organized per EVAL_MATRIX.

A grader takes an already-produced artifact (+ reference) and returns a
GraderResult. It does NOT run the agent — the dataset/fixture layer that feeds
these ("tasks") is separate and added later.

    from eval.graders import scout, lens, quill
    from eval.graders import GraderResult, GraderType

    res = lens.robustness([True, True, False])   # -> GraderResult(pass^k=False)

REGISTRY lists every grader with its kind, so the harness can enumerate/filter
without hard-coding names.
"""

from . import lens, quill, scout
from .base import GraderResult, GraderType, field, judge
from .langsmith_adapter import evaluator, to_langsmith_dict

__all__ = [
    "scout", "lens", "quill", "GraderResult", "GraderType", "field", "judge",
    "evaluator", "to_langsmith_dict", "REGISTRY",
]

# (agent, criterion, kind) -> callable. The kind mirrors the grader's GraderType.
REGISTRY = {
    # Scout
    ("scout", "plan_well_formed", "code"): scout.plan_well_formed,
    ("scout", "target_model_appropriate", "code"): scout.target_model_appropriate,
    ("scout", "citations_exist", "code"): scout.citations_exist,
    ("scout", "tools_called_properly", "code"): scout.tools_called_properly,
    ("scout", "efficiency", "code"): scout.efficiency,
    ("scout", "hypothesis_specificity", "llm"): scout.hypothesis_specificity,
    ("scout", "groundedness", "llm"): scout.groundedness,
    ("scout", "citations_support_hypothesis", "llm"): scout.citations_support_hypothesis,
    ("scout", "search_triggering", "llm"): scout.search_triggering,
    ("scout", "tool_capability_honesty", "llm"): scout.tool_capability_honesty,
    ("scout", "researcher_would_run", "human"): scout.researcher_would_run,
    # Lens — Layer 1 (numeric tools)
    ("lens", "numeric_correctness", "code"): lens.numeric_correctness,
    ("lens", "robustness", "code"): lens.robustness,
    ("lens", "tool_execution_valid", "code"): lens.tool_execution_valid,
    ("lens", "plan_execution_fidelity", "code"): lens.plan_execution_fidelity,
    ("lens", "operational_cost", "code"): lens.operational_cost,
    # Lens — Layer 2 (interpretation)
    ("lens", "prose_data_consistency", "code"): lens.prose_data_consistency,
    ("lens", "interpretation_faithfulness", "llm"): lens.interpretation_faithfulness,
    ("lens", "causal_correlational_honesty", "llm"): lens.causal_correlational_honesty,
    # Quill
    ("quill", "calibration", "code"): quill.calibration,
    ("quill", "false_positive_rate", "code"): quill.false_positive_rate,
    ("quill", "followup_executable", "code"): quill.followup_executable,
    ("quill", "stability", "code"): quill.stability,
    ("quill", "gap_precision", "llm"): quill.gap_precision,
    ("quill", "gap_recall", "llm"): quill.gap_recall,
    ("quill", "false_positives_spurious", "llm"): quill.false_positives_spurious,
    ("quill", "followup_relevant", "llm"): quill.followup_relevant,
    ("quill", "statistical_validity", "llm"): quill.statistical_validity,
    ("quill", "matches_expert", "human"): quill.matches_expert,
}
