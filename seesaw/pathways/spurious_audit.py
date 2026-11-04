"""Spurious audit: does the model's answer depend on something it shouldn't, and can that be removed?

A task the model should do the same way regardless of an attribute
(sentiment shouldn't depend on the reviewer's name; a profession guess
shouldn't depend on the pronoun), and prompt pairs that differ only in that
attribute. The audit asks whether the attribute moves the task's answer,
and whether erasing it fixes that without breaking the task (Marks et al.
2024 did this with SAE features as SHIFT; here it's done with LEACE, so it
works on any model).

Measure → erase → check:

1. Measure — the task works (its logit diff clears TASK_MIN on the
   prompts), and swapping the attribute moves it (reliance: the mean change,
   in logits, across pairs). Under RELIANCE_MIN there's nothing to fix.
2. Erase — fit LEACE erasers on half the pairs, at every residual-stream
   site, so no linear readout could see the attribute; apply them all at
   once on the other half. Random erasers of the same rank are the control.
3. Check — reliance after erasure (should collapse), the task's logit diff
   after erasure (should survive), and loss on control text (the model
   shouldn't be damaged).

Verdicts:
- no_task       — the model doesn't do the task, so reliance means nothing.
- no_reliance   — swapping the attribute doesn't move the answer.
- removable     — erasure removes most of the reliance, far more than random
                  erasure, and the task survives.
- entangled     — erasure removes the reliance but takes the task with it.
- not_removed   — erasure leaves the reliance: it isn't linear, or isn't
                  this attribute.
- inconclusive  — too few pairs to hold any out.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformer_lens import HookedTransformer

from seesaw.interp._helpers import tokens_to_ids
from seesaw.interp.black_box import counterfactual
from seesaw.interp.metrics import last_positions, logit_diff
from seesaw.interp.model_editing import leace
from seesaw.interp.result import InterpResult

from .result import PathwayResult

# The task's logit diff on the prompts must clear this for reliance to mean anything.
TASK_MIN = 0.5
# Mean |change| (logits) from swapping the attribute below which there's no reliance.
RELIANCE_MIN = 0.3
# Erasure must remove this much of the reliance, and beat random erasure by this much...
REMOVE_MIN = 0.5
RANDOM_MARGIN = 0.3
# ...while keeping this share of the task's logit diff and not raising loss past this.
TASK_KEEP_MIN = 0.7
LOSS_MAX_INCREASE = 0.5
MIN_PAIRS = 6


def _lds(model, prompts, pos_ids, neg_ids) -> torch.Tensor:
    with torch.no_grad():
        return logit_diff(model(model.to_tokens(prompts)), pos_ids, neg_ids,
                          last_positions(model, prompts)).cpu()


def _loss(model, texts) -> float:
    with torch.no_grad():
        return sum(model(model.to_tokens(t), return_type="loss").item() for t in texts) / len(texts)


def run(
    model: HookedTransformer,
    prompts: list[str],
    counterfactual_prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    output_dir: Path,
    control_texts: list[str] | None = None,
) -> PathwayResult:
    """Run the spurious audit.

    Args:
        model: Loaded HookedTransformer.
        prompts: Task prompts carrying the attribute (e.g. a review signed "Mary").
        counterfactual_prompts: The same prompts with only the attribute swapped ("John").
        positive_tokens: The task's correct answer per prompt (" positive").
        negative_tokens: The task's wrong answer per prompt (" negative").
        output_dir: Where every step saves its plots.
        control_texts: Ordinary text for the damage check; defaults to the prompts.

    Returns:
        PathwayResult with the task's strength, the reliance before and after
        erasure, the random-erasure control, and a verdict.
    """
    result = PathwayResult(pathway="spurious_audit")
    pos_ids, neg_ids = tokens_to_ids(model, positive_tokens), tokens_to_ids(model, negative_tokens)

    # ── 1. Measure ────────────────────────────────────────────────────────────
    cf = counterfactual.compute(
        model, prompts, counterfactual_prompts, positive_tokens, negative_tokens, output_dir)
    result.steps["counterfactual"] = cf
    task = sum(cf.data["prompt_lds"]) / len(cf.data["prompt_lds"])
    reliance = sum(abs(e) for e in cf.data["effects"]) / len(cf.data["effects"])
    result.findings.update({"task_logit_diff": round(task, 4), "reliance": round(reliance, 4)})
    if task < TASK_MIN:
        return result.stop("no_task", f"the task's logit diff on the prompts is {task:+.2f}, under {TASK_MIN}")
    result.evidence.append(f"The model does the task: it prefers the right answer by {task:+.2f} logits.")
    if reliance < RELIANCE_MIN:
        return result.stop("no_reliance", f"swapping the attribute moves the answer by {reliance:.2f} "
                                          f"logits on average, under {RELIANCE_MIN}")
    result.evidence.append(
        f"Swapping the attribute moves the answer by {reliance:.2f} logits on average "
        f"(net {cf.data['mean_effect']:+.2f}, same direction for {cf.data['sign_consistency']:.0%} of pairs).")
    if len(prompts) < MIN_PAIRS:
        return result.stop("inconclusive", f"{len(prompts)} pairs can't be split into a fitting half "
                                           f"and a held-out half; use at least {MIN_PAIRS}")

    # ── 2. Erase ──────────────────────────────────────────────────────────────
    fit, test = list(range(0, len(prompts), 2)), list(range(1, len(prompts), 2))

    def pick(xs, idx):
        return [xs[i] for i in idx]

    erasers = leace.fit_all(model, pick(prompts, fit), pick(counterfactual_prompts, fit))
    p_test, cf_test = pick(prompts, test), pick(counterfactual_prompts, test)
    pt, nt = pick(pos_ids, test), pick(neg_ids, test)
    texts = control_texts or p_test

    def measure(run_model):
        a, b = _lds(run_model, p_test, pt, nt), _lds(run_model, cf_test, pt, nt)
        return (a - b).abs().mean().item(), a.mean().item(), _loss(run_model, texts)

    before = measure(model)
    with leace.erased(model, erasers):
        after = measure(model)
    with leace.erased(model, leace.random_erasers(erasers)):
        rand = measure(model)

    def removed(m):
        return 1 - m[0] / before[0] if before[0] > 1e-6 else 0.0

    kept = after[1] / before[1] if abs(before[1]) > 1e-6 else 0.0
    loss_up = after[2] - before[2]
    result.steps["leace_erasure"] = InterpResult(data={
        "reliance_before": round(before[0], 4), "reliance_after": round(after[0], 4),
        "reliance_after_random": round(rand[0], 4),
        "task_before": round(before[1], 4), "task_after": round(after[1], 4),
        "control_loss_before": round(before[2], 4), "control_loss_after": round(after[2], 4),
        "random_fraction_removed": round(removed(rand), 4),
    })
    result.findings.update({
        "fraction_removed": round(removed(after), 4),
        "random_fraction_removed": round(removed(rand), 4),
        "task_kept": round(kept, 4),
        "control_loss_increase": round(loss_up, 4),
    })

    # ── 3. Check ──────────────────────────────────────────────────────────────
    result.evidence.append(
        f"On the held-out pairs, erasing the attribute removes {removed(after):.0%} of the reliance "
        f"(random erasers: {removed(rand):.0%}).")
    result.evidence.append(
        f"The task keeps {kept:.0%} of its logit diff ({before[1]:+.2f} → {after[1]:+.2f}); loss on "
        f"control text changes by {loss_up:+.2f} nats.")

    erased_it = removed(after) >= REMOVE_MIN and removed(after) - removed(rand) >= RANDOM_MARGIN
    if erased_it and kept >= TASK_KEEP_MIN and loss_up <= LOSS_MAX_INCREASE:
        result.verdict = "removable"
    elif erased_it:
        result.verdict = "entangled"
    else:
        result.verdict = "not_removed"
    return result
