"""Circuit discovery: nominate heads with DLA, confirm them with ablation, localise with patching.

The standard first pass on a logit-difference task like IOI:

1. Direct logit attribution — which heads write toward the correct answer?
   Descriptive only: a head can write the answer without being necessary.
2. Mean ablation — which heads actually hurt the logit diff when removed?
3. Cross-check — heads that appear in both are the confirmed candidates.
   DLA-only heads are "writes but redundant" (backup / self-repair suspects);
   ablation-only heads are "necessary but indirect" (likely upstream of the
   writers, e.g. S-inhibition heads in IOI).
4. Activation patching (only if corrupted prompts are given) — at which
   layer and token position does the clean signal enter?
"""

from __future__ import annotations

from pathlib import Path

from transformer_lens import HookedTransformer

from seesaw.interp.causal_interventions import ablation, activation_patching
from seesaw.interp.visualization import direct_logit_attribution

from .result import PathwayResult


def run(
    model: HookedTransformer,
    prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    output_dir: Path,
    corrupted_prompts: list[str] | None = None,
    top_k: int = 10,
) -> PathwayResult:
    """Run the circuit-discovery pathway on a logit-difference task.

    Args:
        model:             Loaded HookedTransformer.
        prompts:           Clean prompts.
        positive_tokens:   Correct answer token per prompt (e.g. " Mary").
        negative_tokens:   Distractor token per prompt (e.g. " John").
        output_dir:        Where every step saves its plots.
        corrupted_prompts: Optional counterfactuals; enables the patching step.
        top_k:             How many heads to take from each ranking.

    Returns:
        PathwayResult with each step's InterpResult and the cross-checked heads.
    """
    result = PathwayResult(pathway="circuit_discovery")

    dla = direct_logit_attribution.compute(
        model, prompts, positive_tokens, negative_tokens, output_dir)
    result.steps["direct_logit_attribution"] = dla

    abl = ablation.compute(
        model, prompts, positive_tokens, negative_tokens, output_dir,
        ablation_type="mean")
    result.steps["ablation"] = abl

    # Rank from the full [layer, head] matrices rather than each tool's own
    # top_heads: DLA's is sorted by |attr|, which would lump heads that write
    # *against* the answer (negative name movers in IOI) in with the writers.
    head_attrs = dla.data["head_attrs"]
    effects    = abl.data["effects"]
    dla_pos    = _top(head_attrs, top_k)
    dla_neg    = _top([[-v for v in row] for row in head_attrs], top_k)
    abl_heads  = _top(effects, top_k)

    result.findings = {
        "confirmed_heads":         sorted(dla_pos & abl_heads),
        "writes_but_redundant":    sorted(dla_pos - abl_heads),
        "necessary_but_indirect":  sorted(abl_heads - dla_pos - dla_neg),
        "writes_against_answer":   sorted(dla_neg),
        "baseline_logit_diff":     abl.data["baseline_ld"],
    }

    if corrupted_prompts:
        patch = activation_patching.compute(
            model, prompts, corrupted_prompts,
            positive_tokens, negative_tokens, output_dir)
        result.steps["activation_patching"] = patch
        result.findings["top_recovery_positions"] = patch.data["top_recovery_positions"][:top_k]

    return result


def _top(matrix: list[list[float]], k: int) -> set[str]:
    """The k largest *positive* entries of a [layer][head] matrix, as "L{l}H{h}"."""
    flat = [(v, l, h) for l, row in enumerate(matrix) for h, v in enumerate(row) if v > 0]
    return {f"L{l}H{h}" for _, l, h in sorted(flat, reverse=True)[:k]}
