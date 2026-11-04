"""Editing comparison: is the layer where a fact is traced the right layer to edit it?

ROME edits the MLP where causal tracing says the fact is recalled. Hase et
al. 2023 ("Does Localization Inform Editing?") found that where tracing
points barely predicts which layer gives the best edit. This pathway
reproduces that comparison on your facts and your model.

Trace → edit everywhere → compare:

1. Trace — causal tracing; the subject-last-token MLP effect per layer.
2. Edit — a ROME edit at every layer (each fact separately, weights restored
   after each), scored on efficacy (the edited prompt now says the new
   answer), paraphrase (a rewording does too) and specificity (a different
   subject with the same relation keeps its answer).
3. Compare — the rank correlation between tracing effect and edit score
   across layers, and whether the traced layer is among the best to edit.

Verdicts:
- edits_fail                — no layer's edit works on most facts.
- tracing_predicts_editing  — the traced layer is among the best few to edit,
                              and tracing and editing scores correlate.
- tracing_does_not_predict  — edits work, but not best where tracing points
                              (the Hase et al. finding).
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformer_lens import HookedTransformer

from .result import PathwayResult
from .tools import Tools

# The best layer's edits must work on at least this share of facts.
EFFICACY_MIN = 0.5
# The traced layer must be within this many of the best edit scores...
TOP_K = 3
# ...and tracing and edit score must rank-correlate at least this much.
CORRELATION_MIN = 0.5


def _spearman(a: list[float], b: list[float]) -> float:
    def rank(x):
        return torch.tensor(x).argsort().argsort().float()
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()


def run(
    model: HookedTransformer | str,
    prompts: list[str],
    subjects: list[str],
    positive_tokens: list[str],
    target_tokens: list[str],
    paraphrase_prompts: list[str],
    neighborhood_prompts: list[str],
    neighborhood_tokens: list[str],
    output_dir: Path,
    layers: list[int] | None = None,
) -> PathwayResult:
    """Run the editing comparison.

    Args:
        model: Loaded HookedTransformer, or a model name (see pathways.tools).
        prompts: One factual prompt per fact, ending right before the answer.
        subjects: The subject inside each prompt.
        positive_tokens: The true answer per fact.
        target_tokens: The new answer each edit writes in.
        paraphrase_prompts: A rewording of each prompt (same subject, same answer).
        neighborhood_prompts: A prompt per fact about a different subject with
            the same relation, which the edit shouldn't change.
        neighborhood_tokens: That prompt's true answer.
        output_dir: Where every step saves its plots.
        layers: Layers to edit (default: all).

    Returns:
        PathwayResult with tracing and edit scores per layer, their
        correlation, and a verdict.
    """
    result = PathwayResult(pathway="editing_comparison")
    tools = Tools(model, output_dir)
    layers = list(range(tools.n_layers)) if layers is None else layers

    # ── 1. Trace ──────────────────────────────────────────────────────────────
    ct = tools("causal_tracing", prompts, subjects=subjects, positive_tokens=positive_tokens)
    result.steps["causal_tracing"] = ct
    subj = ct.data["groups"].index("subject_last")
    tracing = [ct.data["effects"]["mlp"][l][subj] for l in layers]
    traced = layers[max(range(len(layers)), key=lambda i: tracing[i])]
    result.findings["traced_layer"] = traced
    result.evidence.append(f"Causal tracing puts the fact in the MLPs around layer {traced} "
                           f"({max(tracing):.0%} of the probability restored).")

    # ── 2. Edit everywhere ────────────────────────────────────────────────────
    # Each fact is edited and scored on its own, and the weights restored, inside the tool.
    edits = tools("rome_edits", prompts, subjects=subjects, positive_tokens=positive_tokens,
                  target_tokens=target_tokens, paraphrase_prompts=paraphrase_prompts,
                  neighborhood_prompts=neighborhood_prompts, neighborhood_tokens=neighborhood_tokens,
                  layers=layers)
    result.steps["rome_edits"] = edits
    per_layer = {l: edits.data["per_layer"][str(l)] for l in layers}
    scores = [per_layer[l]["score"] for l in layers]
    best = layers[max(range(len(layers)), key=lambda i: scores[i])]
    rho = _spearman(tracing, scores)
    ranked = sorted(layers, key=lambda l: -per_layer[l]["score"])
    result.findings.update({
        "edit_scores": {str(l): per_layer[l] for l in layers},
        "best_edit_layer": best,
        "traced_layer_rank": ranked.index(traced) + 1,
        "tracing_edit_correlation": round(rho, 4),
    })
    def describe(s: dict) -> str:
        return ", ".join(f"{k} {s[k]:.0%}" for k in ("efficacy", "paraphrase", "specificity")
                         if s.get(k) is not None)

    result.evidence.append(f"The best layer to edit is {best}: {describe(per_layer[best])}.")
    result.evidence.append(
        f"At the traced layer {traced} (ranked {ranked.index(traced) + 1} of {len(layers)}): "
        f"{describe(per_layer[traced])}.")
    result.evidence.append(f"Tracing effect and edit score rank-correlate at {rho:+.2f} across layers.")

    # ── 3. Compare ────────────────────────────────────────────────────────────
    if max(per_layer[l]["efficacy"] for l in layers) < EFFICACY_MIN:
        result.verdict = "edits_fail"
    elif traced in ranked[:TOP_K] and rho >= CORRELATION_MIN:
        result.verdict = "tracing_predicts_editing"
    else:
        result.verdict = "tracing_does_not_predict"
    return result
