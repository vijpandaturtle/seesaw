"""Concept direction: is a concept a direction the model uses, and can it be steered safely?

The question behind most applied interpretability: refusal, sentiment,
gender, truthfulness. Inputs are prompt pairs that differ only in the
concept ("The nurse said that" / "The engineer said that") and a token pair
the concept might push between (" she" / " he").

The pairs are split in two. The direction is fitted on one half and every
test of it runs on the other, so the direction can't pass by being fitted to
the very prompts it's tested on (a difference of means removes its own
prompts' gap by construction).

Measure → find → test → bake:

1. Measure — counterfactual evaluation on every pair. If swapping the
   concept moves the logit diff by less than EFFECT_MIN, stop.
2. Find (fitting half)
   - Linear probe, with a shuffled-label control: at which layer is the
     concept decodable, beyond what a probe finds in noise? Its
     difference-of-means direction there is the candidate.
   - Attribution patching + head patching (every pair, when they tokenize
     to the same length): which heads carry the effect.
3. Test (held-out half)
   - Directional ablation: remove the direction everywhere. Does the gap
     close, by more than random directions close it, without damaging the
     model's loss on control text?
   - Steering: add the direction to the counterfactuals. Does the behaviour
     move toward the prompts', faster than random directions move it, and
     without the loss jumping?
4. Bake — fold the removal into the weights (no matrix can write the
   direction) and check it matches the hook version, so the edit can ship as
   a checkpoint.

Verdicts:
- no_effect         — swapping the concept doesn't change the output.
- uses_direction    — removing it closes most of the held-out gap, far more
                      than random directions, at little cost.
- partly_uses_direction — removing it closes clearly more of the held-out gap
                      than random directions do, but under half of it; or
                      steering along it moves the behaviour safely. The
                      model uses it alongside something else.
- encoded_not_used  — decodable beyond the shuffled control, but removing it
                      leaves the behaviour: the model gets there another way.
- not_encoded       — no better than shuffled labels, and removal does nothing.
- inconclusive      — anything else (too few pairs, the edit damages the model, ...).
"""

from __future__ import annotations

from pathlib import Path

from transformer_lens import HookedTransformer

from seesaw.interp.result import InterpResult

from .result import PathwayResult
from .tools import Tools

# Mean counterfactual effect (logits) below which there's nothing to study.
EFFECT_MIN = 0.5
# Fewest pairs that can be split into a fitting half and a testing half.
MIN_PAIRS = 6
# A probe must beat its shuffled-label control by this much to count as "encoded".
PROBE_MARGIN = 0.2
# Directional ablation on the held-out half must remove this much of the gap...
REMOVE_MIN = 0.5
# ...beat random directions by this much...
RANDOM_MARGIN = 0.3
# ...and raise control-text loss by no more than this (nats).
LOSS_MAX_INCREASE = 0.5
# Removal beating random directions by this much, short of REMOVE_MIN, is partial use.
PARTIAL_MARGIN = 0.15
# Steering counts when it moves the held-out counterfactuals this share of
# the way to the prompts, with a slope this many times the random directions'.
STEER_MIN = 0.5
STEER_OVER_RANDOM = 2.0
# Baked weights must reproduce the hook's held-out effect to within this share of the gap.
BAKE_TOLERANCE = 0.05
SHORTLIST_K = 10
CARRIER_MIN = 0.10


def run(
    model: HookedTransformer | str,
    prompts: list[str],
    counterfactual_prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    output_dir: Path,
    control_prompts: list[str] | None = None,
    control_texts: list[str] | None = None,
    pathway: str = "concept_direction",
) -> PathwayResult:
    """Run the concept-direction pathway.

    Args:
        model: Loaded HookedTransformer, or a model name (see pathways.tools).
        prompts: Prompts carrying the concept (e.g. female-stereotyped occupations).
        counterfactual_prompts: The same prompts with the concept swapped.
        positive_tokens: Token per prompt the concept might push toward (" she").
        negative_tokens: Token per prompt it might push away from (" he").
        output_dir: Where every step saves its plots.
        control_prompts: Concept-free prompts ("The person said that"), to
            show the model's baseline preference.
        control_texts: Ordinary text for checking the edit doesn't damage the
            model; defaults to the prompts.
        pathway: Name recorded on the result (attribute_audit reuses this pathway).

    Returns:
        PathwayResult with every step, the direction's layer, carrier heads,
        held-out removal and steering, the bake check, and a verdict.
    """
    result = PathwayResult(pathway=pathway)
    tools = Tools(model, output_dir)

    # ── 1. Measure ────────────────────────────────────────────────────────────
    cf = tools("counterfactual", prompts, counterfactual_prompts=counterfactual_prompts,
               positive_tokens=positive_tokens, negative_tokens=negative_tokens,
               control_prompts=control_prompts)
    result.steps["counterfactual"] = cf
    effect = cf.data["mean_effect"]
    result.findings.update({"mean_effect": effect, "sign_consistency": cf.data["sign_consistency"]})
    if abs(effect) < EFFECT_MIN:
        return result.stop(
            "no_effect", f"swapping the concept moves the logit diff by {effect:+.2f}, under {EFFECT_MIN}")
    result.evidence.append(
        f"Swapping the concept moves logit({positive_tokens[0]!r}) - logit({negative_tokens[0]!r}) "
        f"by {effect:+.2f} on average, in the same direction for {cf.data['sign_consistency']:.0%} of pairs.")
    if cf.data["control_mean_ld"] is not None:
        result.evidence.append(f"Concept-free control prompts sit at {cf.data['control_mean_ld']:+.2f}.")
    if len(prompts) < MIN_PAIRS:
        return result.stop("inconclusive", f"{len(prompts)} pairs can't be split into a fitting half "
                                           f"and a held-out half; use at least {MIN_PAIRS}")

    fit, test = list(range(0, len(prompts), 2)), list(range(1, len(prompts), 2))

    def pick(xs, idx):
        return [xs[i] for i in idx]

    p_fit, cf_fit = pick(prompts, fit), pick(counterfactual_prompts, fit)
    p_test, cf_test = pick(prompts, test), pick(counterfactual_prompts, test)
    pos_test, neg_test = pick(positive_tokens, test), pick(negative_tokens, test)
    result.findings["split"] = {"fit": len(fit), "held_out": len(test)}

    # ── 2. Find ───────────────────────────────────────────────────────────────
    probe = tools("linear_probe", p_fit, counterfactual_prompts=cf_fit)
    result.steps["linear_probe"] = probe
    layer, acc = probe.data["best_layer"], probe.data["best_accuracy"]
    shuffled = probe.data["shuffled_accuracy"]
    encoded = acc - shuffled >= PROBE_MARGIN
    # The direction itself (the difference of means at `layer` on the fitting
    # half) is rebuilt from the same prompts by each tool that uses it.
    fitted = {"layer": layer, "fit_prompts": p_fit, "fit_counterfactual_prompts": cf_fit}
    result.findings.update({"probe_layer": layer, "probe_accuracy": acc, "shuffled_accuracy": shuffled})
    result.evidence.append(
        f"On the fitting half, a probe separates the two groups with {acc:.0%} accuracy from layer "
        f"{layer}; on shuffled labels it scores {shuffled:.0%}"
        + (" — no better, so nothing here is encoded beyond noise." if not encoded else "."))

    # Padded batch shapes, as the patching tools see them.
    aligned = max(tools.token_lengths(prompts)) == max(tools.token_lengths(counterfactual_prompts))
    if aligned:
        patch = {"corrupted_prompts": counterfactual_prompts, "positive_tokens": positive_tokens,
                 "negative_tokens": negative_tokens}
        atp = tools("attribution_patching", prompts, **patch, top_k=SHORTLIST_K)
        result.steps["attribution_patching"] = atp
        shortlist = [h for h, _ in atp.data["top_heads"]]
        hp = tools("head_patching", prompts, **patch, heads=shortlist)
        result.steps["head_patching"] = hp
        carriers = {h: r for h, r in hp.data["per_head"].items() if r >= CARRIER_MIN}
        result.findings["carrier_heads"] = carriers
        if carriers:
            top = sorted(carriers.items(), key=lambda kv: -kv[1])[:4]
            result.evidence.append(
                "Heads carrying the effect (fraction restored when patched alone): "
                + ", ".join(f"{h} {r:.0%}" for h, r in top) + ".")
    else:
        result.findings["carrier_heads"] = None
        result.evidence.append("Prompt pairs differ in length, so head-level patching was skipped.")

    # ── 3. Test, on the held-out half ─────────────────────────────────────────
    held_out = {"counterfactual_prompts": cf_test, "positive_tokens": pos_test,
                "negative_tokens": neg_test, "control_texts": control_texts, **fitted}
    da = tools("directional_ablation", p_test, **held_out, bake=True)
    result.steps["directional_ablation"] = da
    removed, rand = da.data["fraction_removed"], da.data["random_fraction_removed"]
    loss_up = da.data["control_loss_after"] - da.data["control_loss_before"]
    held_gap = da.data["effect_before"]
    result.findings.update({
        "fraction_removed": removed, "random_fraction_removed": rand,
        "control_loss_increase": round(loss_up, 4),
    })
    result.evidence.append(
        f"On the held-out half, removing the direction everywhere closes {removed:.0%} of the "
        f"{held_gap:+.2f} gap (random directions: {rand:.0%}); loss on control text changes by "
        f"{loss_up:+.2f} nats.")

    steer = tools("steering_vector", p_test, **held_out, alphas=[0.0, 0.25, 0.5, 0.75, 1.0])
    result.steps["steering"] = steer
    shift = steer.data["lds"][-1] - steer.data["lds"][0]
    moved = shift / held_gap if abs(held_gap) > 1e-6 else 0.0
    over_random = abs(steer.data["slope"]) / max(steer.data["random_abs_slope"] or 0.0, 1e-6)
    steer_loss = steer.data["control_loss_after"] - steer.data["control_loss_before"]
    steerable = (moved >= STEER_MIN and over_random >= STEER_OVER_RANDOM
                 and steer_loss <= LOSS_MAX_INCREASE)
    result.findings.update({
        "steering_moved": round(moved, 4), "steering_over_random": round(over_random, 2),
        "steering_loss_increase": round(steer_loss, 4), "steerable": steerable,
    })
    result.evidence.append(
        f"Adding the direction to the held-out counterfactuals moves them {moved:.0%} of the way to "
        f"the prompts, {over_random:.1f}× as fast as random directions, with loss "
        f"{steer_loss:+.2f} nats.")

    # ── 4. Bake ───────────────────────────────────────────────────────────────
    # directional_ablation(bake=True) folded the removal into the weights,
    # re-measured, and restored them.
    baked = da.data["baked_effect_after"]
    hooked = da.data["effect_after"]
    matches = abs(baked - hooked) <= BAKE_TOLERANCE * abs(held_gap) + 1e-3
    result.steps["bake_check"] = InterpResult(data={
        "baked_gap": round(baked, 4), "hooked_gap": hooked, "matches": matches})
    result.findings["bakeable"] = matches
    result.evidence.append(
        f"Folded into the weights, the edit leaves a {baked:+.2f} gap against the hook's {hooked:+.2f}"
        + (": it can ship as a checkpoint." if matches else ": they disagree, so don't ship the baked weights."))

    # ── Conclude ──────────────────────────────────────────────────────────────
    if removed >= REMOVE_MIN and removed - rand >= RANDOM_MARGIN and loss_up <= LOSS_MAX_INCREASE:
        result.verdict = "uses_direction"
    elif (removed - rand >= PARTIAL_MARGIN and loss_up <= LOSS_MAX_INCREASE) or steerable:
        result.verdict = "partly_uses_direction"
    elif encoded and removed < REMOVE_MIN:
        result.verdict = "encoded_not_used"
    elif not encoded and removed < REMOVE_MIN:
        result.verdict = "not_encoded"
    else:
        result.verdict = "inconclusive"
    return result
