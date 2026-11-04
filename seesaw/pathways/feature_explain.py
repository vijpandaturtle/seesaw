"""Feature explanation: which interpretable SAE features drive this output, and what are they?

At scale, single heads stop being the right unit: their outputs mix many
unrelated signals. Sparse autoencoder features are the unit current interp
work uses instead. This pathway finds the features behind a logit-difference
behaviour at one SAE's hook point and labels them with Neuronpedia's
explanations.

One SAE reads one layer, and a behaviour can be invisible at the wrong one,
so the pathway takes several SAEs (one per layer), runs nominate and verify
at each, and explains the layer where switching features off accounts for
the most.

Measure → nominate → explain → verify:

1. Measure — logit diff on the prompts must clear EFFECT_MIN.
2. Nominate — feature attribution (activation × gradient) over every feature.
3. Explain — Neuronpedia explanations for the top features, plus (optional)
   Neuronpedia's server-side search for what fires on the prompts at all, as
   context. Explanations are auto-generated and can be wrong; they're labels
   for a human to check, not evidence.
4. Verify — clamp the top features to zero, one at a time and together,
   against random active features. (Nominate and verify run at every SAE;
   explain runs at the best one.)

Verdicts: no_effect · features_explain · partial · not_explained
"""

from __future__ import annotations

from pathlib import Path

from transformer_lens import HookedTransformer

from seesaw.interp.feature_extraction import feature_attribution, feature_search, neuronpedia
from seesaw.interp.feature_extraction.sae import hook_name, neuronpedia_source
from seesaw.interp.steering import feature_clamping

from .result import PathwayResult

# Baseline logit diff (logits) below which there's nothing to explain.
EFFECT_MIN = 1.0
# Features passed from attribution to clamping.
TOP_K = 8
# Clamping the top features together must remove this much of the logit diff...
EXPLAIN_MIN = 0.5
# ...and beat random active features by this much.
RANDOM_MARGIN = 0.3
# One feature alone removing this much counts toward "partial".
SINGLE_MIN = 0.1


def _at_layer(model, sae, prompts, positive_tokens, negative_tokens, output_dir, result, tag):
    """Nominate and verify at one SAE. Returns its numbers, or None when nothing is nominated."""
    fa = feature_attribution.compute(
        model, sae, prompts, positive_tokens, negative_tokens, output_dir, top_k=TOP_K)
    result.steps[f"feature_attribution{tag}"] = fa
    baseline, top = fa.data["baseline_ld"], fa.data["top_features"]
    if baseline < EFFECT_MIN or not top:
        return {"fa": fa, "baseline": baseline, "top": top}
    indices = [f["index"] for f in top]
    fc = feature_clamping.compute(
        model, sae, prompts, positive_tokens, negative_tokens, indices, output_dir)
    result.steps[f"feature_clamping{tag}"] = fc
    rand = fc.data["random_joint_change"]
    return {
        "fa": fa, "fc": fc, "baseline": baseline, "top": top, "sae": sae,
        "explained": -fc.data["joint_change"] / baseline,
        "rand_explained": -rand / baseline if rand is not None else None,
        "per_feature": {int(k): -v / baseline for k, v in fc.data["per_feature_change"].items()},
    }


def run(
    model: HookedTransformer,
    sae,
    prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    output_dir: Path,
    search_neuronpedia: bool = True,
) -> PathwayResult:
    """Run the feature-explanation pathway.

    Args:
        model: Loaded HookedTransformer.
        sae: An SAE on one of the model's hook points, or a list of them (one
            per layer) to sweep (see feature_extraction.sae.load_sae).
        prompts: Prompts the behaviour shows up on.
        positive_tokens: Correct answer token per prompt.
        negative_tokens: Distractor token per prompt.
        output_dir: Where every step saves its plots.
        search_neuronpedia: Also ask Neuronpedia which features fire on the
            prompts at the best layer (context only; skipped if the SAE isn't
            hosted there or the request fails).

    Returns:
        PathwayResult with each layer's explained share, the best layer's top
        features, their labels, clamping effects and a verdict.
    """
    result = PathwayResult(pathway="feature_explain")
    saes = list(sae) if isinstance(sae, (list, tuple)) else [sae]

    # ── 1-2 and 4. Measure, nominate and verify at every SAE ──────────────────
    runs = []
    for one in saes:
        tag = f"@{hook_name(one)}" if len(saes) > 1 else ""
        runs.append(_at_layer(model, one, prompts, positive_tokens, negative_tokens, output_dir, result, tag))
    baseline = runs[0]["baseline"]
    result.findings["baseline_logit_diff"] = baseline
    if baseline < EFFECT_MIN:
        return result.stop("no_effect", f"logit diff {baseline:.2f} is below {EFFECT_MIN}, nothing to explain")
    verified = [r for r in runs if "explained" in r]
    if not verified:
        return result.stop("not_explained", "no feature has positive attribution at any layer")
    if len(saes) > 1:
        result.findings["explained_by_layer"] = {
            hook_name(r["sae"]): round(r["explained"], 4) for r in verified}
    best = max(verified, key=lambda r: r["explained"])
    hook = best["fa"].data["hook_name"]
    top, explained, per_feature = best["top"], best["explained"], best["per_feature"]
    rand_explained = best["rand_explained"]
    result.evidence.append(
        f"The model prefers the answer by {baseline:.2f} logits; attribution ranks "
        f"{len(top)} features at {hook} as the main contributors"
        + (f" (the best of {len(saes)} layers)." if len(saes) > 1 else "."))

    # ── 3. Explain ────────────────────────────────────────────────────────────
    result.findings["top_features"] = [
        {k: f.get(k) for k in ("index", "attribution", "explanation", "url")} for f in top
    ]
    source = neuronpedia_source(best["sae"])
    if search_neuronpedia and source:
        model_id, sae_id = source.split("/", 1)
        source_set = sae_id.split("-", 1)[1] if "-" in sae_id else sae_id
        try:
            fs = feature_search.compute(prompts, model_id, source_set, [sae_id], output_dir, top_k=10)
            result.steps["feature_search"] = fs
            result.findings["most_active_features"] = [
                {k: f[k] for k in ("index", "explanation", "fires_on")} for f in fs.data["features"][:5]
            ]
        except neuronpedia.NeuronpediaError as exc:
            result.evidence.append(f"Neuronpedia search skipped: {exc}.")

    # ── Conclude ──────────────────────────────────────────────────────────────
    result.findings.update({
        "layer": hook,
        "explained_fraction": round(explained, 4),
        "random_explained_fraction": None if rand_explained is None else round(rand_explained, 4),
        "per_feature_fraction": {k: round(v, 4) for k, v in per_feature.items()},
    })
    labels = {f["index"]: f.get("explanation") for f in top}
    for idx, frac in sorted(per_feature.items(), key=lambda kv: -kv[1])[:4]:
        label = f" (\"{labels[idx]}\")" if labels.get(idx) else ""
        result.evidence.append(f"Feature #{idx}{label}: switching it off removes {frac:.0%} of the logit diff.")
    if rand_explained is None:
        result.evidence.append(
            f"Switching off all {len(top)} together removes {explained:.0%}; there were too few "
            "active features to draw a random comparison, so this can't be called explained.")
    else:
        result.evidence.append(
            f"Switching off all {len(top)} together removes {explained:.0%}; "
            f"the same number of random active features removes {rand_explained:.0%}.")

    if (rand_explained is not None and explained >= EXPLAIN_MIN
            and explained - rand_explained >= RANDOM_MARGIN):
        result.verdict = "features_explain"
    elif any(v >= SINGLE_MIN for v in per_feature.values()):
        result.verdict = "partial"
    else:
        result.verdict = "not_explained"
    return result
