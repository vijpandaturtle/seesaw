"""Feature explanation: which interpretable SAE features drive this output, and what are they?

At scale, single heads stop being the right unit: their outputs mix many
unrelated signals. Sparse autoencoder features are the unit current interp
work uses instead. This pathway finds the features behind a logit-difference
behaviour at one SAE's hook point and labels them with Neuronpedia's
explanations.

Measure → nominate → explain → verify:

1. Measure — logit diff on the prompts must clear EFFECT_MIN.
2. Nominate — feature attribution (activation × gradient) over every feature.
3. Explain — Neuronpedia explanations for the top features, plus (optional)
   Neuronpedia's server-side search for what fires on the prompts at all, as
   context. Explanations are auto-generated and can be wrong; they're labels
   for a human to check, not evidence.
4. Verify — clamp the top features to zero, one at a time and together,
   against random active features.

Verdicts: no_effect · features_explain · partial · not_explained
"""

from __future__ import annotations

from pathlib import Path

from transformer_lens import HookedTransformer

from seesaw.interp.feature_extraction import feature_attribution, feature_search, neuronpedia
from seesaw.interp.feature_extraction.sae import neuronpedia_source
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
        sae: SAE on one of the model's hook points (see feature_extraction.sae.load_sae).
        prompts: Prompts the behaviour shows up on.
        positive_tokens: Correct answer token per prompt.
        negative_tokens: Distractor token per prompt.
        output_dir: Where every step saves its plots.
        search_neuronpedia: Also ask Neuronpedia which features fire on the
            prompts (context only; skipped if the SAE isn't hosted there or
            the request fails).

    Returns:
        PathwayResult with the top features, their labels, clamping effects and a verdict.
    """
    result = PathwayResult(pathway="feature_explain")

    # ── 1-2. Measure and nominate (one forward + backward pass) ──────────────
    fa = feature_attribution.compute(
        model, sae, prompts, positive_tokens, negative_tokens, output_dir, top_k=TOP_K)
    result.steps["feature_attribution"] = fa
    baseline = fa.data["baseline_ld"]
    result.findings["baseline_logit_diff"] = baseline
    if baseline < EFFECT_MIN:
        return result.stop("no_effect", f"logit diff {baseline:.2f} is below {EFFECT_MIN}, nothing to explain")
    top = fa.data["top_features"]
    if not top:
        return result.stop("not_explained", "no feature has positive attribution")
    result.evidence.append(
        f"The model prefers the answer by {baseline:.2f} logits; attribution ranks "
        f"{len(top)} features at {fa.data['hook_name']} as the main contributors.")

    # ── 3. Explain ────────────────────────────────────────────────────────────
    result.findings["top_features"] = [
        {k: f.get(k) for k in ("index", "attribution", "explanation", "url")} for f in top
    ]
    source = neuronpedia_source(sae)
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

    # ── 4. Verify ─────────────────────────────────────────────────────────────
    indices = [f["index"] for f in top]
    fc = feature_clamping.compute(
        model, sae, prompts, positive_tokens, negative_tokens, indices, output_dir)
    result.steps["feature_clamping"] = fc
    explained = -fc.data["joint_change"] / baseline
    rand = fc.data["random_joint_change"]
    rand_explained = -rand / baseline if rand is not None else 0.0
    per_feature = {int(k): -v / baseline for k, v in fc.data["per_feature_change"].items()}
    result.findings.update({
        "explained_fraction": round(explained, 4),
        "random_explained_fraction": round(rand_explained, 4),
        "per_feature_fraction": {k: round(v, 4) for k, v in per_feature.items()},
    })
    labels = {f["index"]: f.get("explanation") for f in top}
    for idx, frac in sorted(per_feature.items(), key=lambda kv: -kv[1])[:4]:
        label = f" (\"{labels[idx]}\")" if labels.get(idx) else ""
        result.evidence.append(f"Feature #{idx}{label}: switching it off removes {frac:.0%} of the logit diff.")
    result.evidence.append(
        f"Switching off all {len(indices)} together removes {explained:.0%}; "
        f"the same number of random active features removes {rand_explained:.0%}.")

    if explained >= EXPLAIN_MIN and explained - rand_explained >= RANDOM_MARGIN:
        result.verdict = "features_explain"
    elif any(v >= SINGLE_MIN for v in per_feature.values()):
        result.verdict = "partial"
    else:
        result.verdict = "not_explained"
    return result
