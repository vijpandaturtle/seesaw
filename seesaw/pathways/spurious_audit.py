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

from transformer_lens import HookedTransformer

from .result import PathwayResult
from .tools import Tools

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


def run(
    model: HookedTransformer | str,
    prompts: list[str],
    counterfactual_prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    output_dir: Path,
    control_texts: list[str] | None = None,
) -> PathwayResult:
    """Run the spurious audit.

    Args:
        model: Loaded HookedTransformer, or a model name (see pathways.tools).
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
    tools = Tools(model, output_dir)

    # ── 1. Measure ────────────────────────────────────────────────────────────
    cf = tools("counterfactual", prompts, counterfactual_prompts=counterfactual_prompts,
               positive_tokens=positive_tokens, negative_tokens=negative_tokens)
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

    # leace_erasure fits the erasers on the fitting half, applies them (and
    # random ones) on the held-out half, and measures before and after.
    le = tools("leace_erasure", pick(prompts, test),
               counterfactual_prompts=pick(counterfactual_prompts, test),
               positive_tokens=pick(positive_tokens, test), negative_tokens=pick(negative_tokens, test),
               fit_prompts=pick(prompts, fit), fit_counterfactual_prompts=pick(counterfactual_prompts, fit),
               control_texts=control_texts)
    result.steps["leace_erasure"] = le
    removed, rand_removed = le.data["fraction_removed"], le.data["random_fraction_removed"]
    kept, before_task, after_task = le.data["task_kept"], le.data["task_before"], le.data["task_after"]
    loss_up = le.data["control_loss_after"] - le.data["control_loss_before"]
    result.findings.update({
        "fraction_removed": removed,
        "random_fraction_removed": rand_removed,
        "task_kept": kept,
        "control_loss_increase": round(loss_up, 4),
    })

    # ── 3. Check ──────────────────────────────────────────────────────────────
    result.evidence.append(
        f"On the held-out pairs, erasing the attribute removes {removed:.0%} of the reliance "
        f"(random erasers: {rand_removed:.0%}).")
    result.evidence.append(
        f"The task keeps {kept:.0%} of its logit diff ({before_task:+.2f} → {after_task:+.2f}); loss on "
        f"control text changes by {loss_up:+.2f} nats.")

    erased_it = removed >= REMOVE_MIN and removed - rand_removed >= RANDOM_MARGIN
    if erased_it and kept >= TASK_KEEP_MIN and loss_up <= LOSS_MAX_INCREASE:
        result.verdict = "removable"
    elif erased_it:
        result.verdict = "entangled"
    else:
        result.verdict = "not_removed"
    return result
