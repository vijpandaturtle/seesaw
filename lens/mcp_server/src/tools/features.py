"""Agent-facing wrappers over the SAE feature tools: feature_search, feature_attribution, feature_clamping,
feature_steering, feature_lookup.

The agent names a layer; the SAE comes from SAELens's directory
(seesaw.interp.feature_extraction.sae.find_sae), preferring the release
Neuronpedia hosts so features come back with explanations. Every tool here
defaults to the same layer, so a feature index from feature_attribution
means the same feature when feature_clamping gets it.
"""

from pathlib import Path

from transformer_lens import HookedTransformer

from seesaw.interp.feature_extraction import feature_attribution as _feature_attribution
from seesaw.interp.feature_extraction import feature_search as _feature_search
from seesaw.interp.feature_extraction import neuronpedia as _neuronpedia
from seesaw.interp.feature_extraction.sae import find_sae, load_sae
from seesaw.interp.steering import feature_clamping as _feature_clamping
from seesaw.interp.steering import feature_steering as _feature_steering

from ..config import PLOTS_DIR
from ..models.schemas import ExperimentResult
from ._wrap import wrap


def _default_layer(model: HookedTransformer) -> int:
    # Two-thirds of the way up: past token-level features, before the final
    # layers that mostly map to the output vocabulary.
    return (2 * model.cfg.n_layers) // 3


def _sae(model: HookedTransformer, layer: int | None):
    layer = _default_layer(model) if layer is None else layer
    release, sae_id, source = find_sae(model.cfg.model_name, layer)
    return layer, release, sae_id, source


def _load(model: HookedTransformer, layer: int | None):
    """(layer, loaded SAE) for this model at `layer`: the SAE every feature tool here reads."""
    layer, release, sae_id, _ = _sae(model, layer)
    return layer, load_sae(release, sae_id, device=str(model.cfg.device))


def run_feature_search(
    model: HookedTransformer,
    prompts: list[str],
    layer: int | None = None,
    top_k: int = 15,
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """List the SAE features that fire most across the prompts, with Neuronpedia's explanations.

    Runs on Neuronpedia's servers. Descriptive: firing isn't the same as mattering.
    Answers: "What concepts does the model represent while reading these prompts?"
    """
    layer, _, _, source = _sae(model, layer)
    if source is None:
        raise ValueError(f"Neuronpedia doesn't host the SAE for {model.cfg.model_name} layer {layer}")
    model_id, sae_id = source.split("/", 1)
    source_set = sae_id.split("-", 1)[1]
    result = _feature_search.compute(prompts, model_id, source_set, [sae_id], output_dir, top_k=top_k)
    result.data["layer"] = layer
    return wrap(result, f"Feature Search (layer {layer})", "feature_search", model, prompts)


def run_feature_attribution(
    model: HookedTransformer,
    prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    layer: int | None = None,
    top_k: int = 10,
    explain: bool = True,
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Score every SAE feature at `layer` by its contribution to the logit diff, labelled from Neuronpedia.

    An estimate: confirm the top features with feature_clamping.
    Answers: "Which interpretable features drive this output?"
    """
    layer, sae = _load(model, layer)
    result = _feature_attribution.compute(model, sae, prompts, positive_tokens, negative_tokens,
                                          output_dir, top_k=top_k, explain=explain)
    result.data["layer"] = layer
    return wrap(result, f"Feature Attribution (layer {layer})", "feature_attribution", model, prompts)


def run_feature_clamping(
    model: HookedTransformer,
    prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    features: list[int],
    layer: int | None = None,
    value: float = 0.0,
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Set the listed SAE features to `value` (0 = off) and measure the logit diff, vs random features.

    `layer` must be the one the feature indices came from.
    Answers: "Does the output actually depend on these features?"
    """
    layer, sae = _load(model, layer)
    result = _feature_clamping.compute(model, sae, prompts, positive_tokens, negative_tokens,
                                       [int(f) for f in features], output_dir, value=value)
    result.data["layer"] = layer
    return wrap(result, f"Feature Clamping (layer {layer})", "feature_clamping", model, prompts)


def run_feature_steering(
    model: HookedTransformer,
    prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    feature: int,
    layer: int | None = None,
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Push the residual stream along one SAE feature's decoder direction, sweeping its strength.

    Strength is in multiples of the residual's own size; compared with random
    directions pushed the same way, with a loss check. `layer` must be the one
    the feature index came from.
    Answers: "Is turning this feature up enough to change the output?"
    """
    layer, sae = _load(model, layer)
    result = _feature_steering.compute(model, sae, prompts, positive_tokens, negative_tokens,
                                       int(feature), output_dir)
    result.data["layer"] = layer
    return wrap(result, f"Feature Steering (#{int(feature)}, layer {layer})", "feature_steering",
                model, prompts)


def run_feature_lookup(
    model: HookedTransformer,
    prompts: list[str],
    features: list[int],
    layer: int | None = None,
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """Look the listed SAE features up on Neuronpedia: explanation, tokens they promote and suppress, dashboard link.

    `layer` must be the one the indices came from. The prompts are not used.
    Explanations are written automatically and can be wrong: labels, not evidence.
    Answers: "What are these features?"
    """
    layer, _, _, source = _sae(model, layer)
    looked_up = []
    for index in (int(f) for f in features):
        entry = {"index": index, "explanation": None,
                 "url": _neuronpedia.feature_url(source, index) if source else None}
        if source:
            try:
                entry.update(_neuronpedia.get_feature(source, index))
            except _neuronpedia.NeuronpediaError as exc:
                entry["error"] = str(exc)
        looked_up.append(entry)
    return ExperimentResult(
        name=f"Feature Lookup (layer {layer})", tool="feature_lookup",
        model_name=model.cfg.model_name, prompts=prompts,
        data={"layer": layer, "neuronpedia_source": source, "features": looked_up},
        status="success",
    )
