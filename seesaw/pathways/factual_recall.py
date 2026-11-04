"""Factual recall: where does the model retrieve a fact, and does it follow the ROME pattern?

Meng et al. 2022 found that facts about a subject ("The Eiffel Tower is
located in the city of" → " Paris") are recalled by MLPs at the subject's
last token in early-middle layers, and moved to the answer by attention at
the final token in later layers. This pathway traces a set of facts and says
whether the model does the same.

Measure → trace → conclude:

1. Measure — the model must know the facts: the true answer's probability
   averages at least P_MIN.
2. Trace — causal tracing: corrupt the subject's embeddings with noise, then
   restore the residual stream, MLP outputs or attention outputs at each
   layer and position group, and see how much probability comes back.
3. Conclude — compare the MLP effect at the subject's last token with the
   MLP effect at the final token, and where each kind peaks.

Verdicts:
- no_recall            — the model doesn't know the facts.
- corruption_too_weak  — noising the subject barely changes the answer, so
                         there's nothing to restore.
- subject_mlp_recall   — the ROME pattern: MLPs at the subject's last token,
                         before the final layers, restore much more than MLPs
                         at the final token.
- late_site            — the strongest MLP effect is at the final token.
- not_localized        — no MLP window restores enough on its own.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformer_lens import HookedTransformer

from seesaw.interp._helpers import tokens_to_ids
from seesaw.interp.causal_interventions import causal_tracing

from .result import PathwayResult

# Mean probability of the true answer the model must reach to "know" the facts.
P_MIN = 0.05
# Corruption must cut the answer's probability by at least this share of it.
CORRUPTION_MIN = 0.3
# An MLP window must restore this share of the probability to count as a site.
RESTORE_MIN = 0.15
# The subject-token MLP peak must beat the final-token MLP peak by this factor.
SUBJECT_OVER_FINAL = 1.5


def run(
    model: HookedTransformer,
    prompts: list[str],
    subjects: list[str],
    positive_tokens: list[str],
    output_dir: Path,
) -> PathwayResult:
    """Run the factual-recall pathway.

    Args:
        model: Loaded HookedTransformer.
        prompts: Factual prompts, each ending right before the answer.
        subjects: The subject inside each prompt (must appear verbatim).
        positive_tokens: The true answer token per prompt.
        output_dir: Where every step saves its plots.

    Returns:
        PathwayResult with the tracing grids, each kind's peak, and a verdict.
    """
    result = PathwayResult(pathway="factual_recall")

    # ── 1. Measure ────────────────────────────────────────────────────────────
    ids = tokens_to_ids(model, positive_tokens)
    probs = []
    for p, i in zip(prompts, ids):
        with torch.no_grad():
            probs.append(model(model.to_tokens(p))[0, -1].softmax(-1)[i].item())
    known = sum(probs) / len(probs)
    result.findings["answer_probability"] = round(known, 4)
    if known < P_MIN:
        return result.stop("no_recall", f"the true answers average probability {known:.3f}, under {P_MIN}")
    result.evidence.append(f"The model gives the true answers {known:.0%} probability on average.")

    # ── 2. Trace ──────────────────────────────────────────────────────────────
    ct = causal_tracing.compute(model, prompts, subjects, positive_tokens, output_dir)
    result.steps["causal_tracing"] = ct
    clean, corrupt = ct.data["clean_prob"], ct.data["corrupted_prob"]
    result.findings.update({"corrupted_probability": corrupt, "peaks": ct.data["peaks"]})
    if clean <= 0 or (clean - corrupt) / clean < CORRUPTION_MIN:
        return result.stop("corruption_too_weak", f"noising the subject only takes the answer from "
                                                  f"{clean:.0%} to {corrupt:.0%}")
    result.evidence.append(f"Noising the subject's embeddings drops the answer from {clean:.0%} to {corrupt:.0%}.")

    # ── 3. Conclude ───────────────────────────────────────────────────────────
    groups = ct.data["groups"]
    mlp = torch.tensor(ct.data["effects"]["mlp"])
    attn = torch.tensor(ct.data["effects"]["attn"])
    subj, final = groups.index("subject_last"), groups.index("final")
    subj_layer = int(mlp[:, subj].argmax())
    subj_peak, final_peak = mlp[subj_layer, subj].item(), mlp[:, final].max().item()
    attn_layer = int(attn[:, final].argmax())
    n_layers = model.cfg.n_layers
    result.findings.update({
        "subject_mlp_layer": subj_layer, "subject_mlp_recovery": round(subj_peak, 4),
        "final_mlp_recovery": round(final_peak, 4),
        "final_attention_layer": attn_layer,
        "final_attention_recovery": round(attn[attn_layer, final].item(), 4),
    })
    result.evidence.append(
        f"MLPs around layer {subj_layer} at the subject's last token restore {subj_peak:.0%} of the "
        f"probability; the best MLP window at the final token restores {final_peak:.0%}.")
    result.evidence.append(
        f"Attention at the final token matters most around layer {attn_layer} "
        f"({attn[attn_layer, final].item():.0%}).")

    if max(subj_peak, final_peak) < RESTORE_MIN:
        result.verdict = "not_localized"
    elif subj_peak >= SUBJECT_OVER_FINAL * final_peak and subj_layer < n_layers - 2:
        result.verdict = "subject_mlp_recall"
    elif final_peak > subj_peak:
        result.verdict = "late_site"
    else:
        result.verdict = "not_localized"
    return result
