"""Feature search: which SAE features fire on these prompts, according to Neuronpedia?

Runs entirely on Neuronpedia's servers — no model, no SAE download — so it
works for models far too large to load here, and costs one HTTP request per
prompt. Features are aggregated across prompts: one that fires on every
prompt is more likely about the shared behaviour than about a single
sentence's wording.

Descriptive only, like the logit lens: a feature firing doesn't mean the
output depends on it. `feature_attribution` + `feature_clamping` test that.

Answers: "What concepts does the model represent while reading these prompts?"
"""

from pathlib import Path

from ..result import InterpResult
from . import neuronpedia


def compute(
    prompts: list[str],
    model_id: str,
    source_set: str,
    layers: list[str],
    output_dir: Path,
    top_k: int = 15,
    per_prompt: int = 30,
) -> InterpResult:
    """Aggregate the top-firing Neuronpedia features over a set of prompts.

    Args:
        prompts: Texts to run.
        model_id: Neuronpedia model id, e.g. 'gpt2-small', 'gemma-2-2b'.
        source_set: SAE family, e.g. 'res-jb', 'gemmascope-res-16k'.
        layers: SAE ids within the family, e.g. ['8-res-jb'].
        output_dir: Where the raw per-prompt results are written (JSON).
        top_k: Features to report after aggregating.
        per_prompt: Features requested per prompt before aggregating.

    Returns:
        InterpResult with features ranked by how many prompts they fire on,
        then by mean peak activation.
    """
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    by_feature: dict[tuple[str, int], dict] = {}
    raw = []
    for prompt in prompts:
        result = neuronpedia.search_text(model_id, source_set, layers, prompt, n=per_prompt)
        raw.append({"prompt": prompt, **result})
        for f in result["features"]:
            key = (f["source"], f["index"])
            entry = by_feature.setdefault(key, {
                "source": f["source"], "index": f["index"], "explanation": f["explanation"],
                "url": f["url"], "n_prompts": 0, "peaks": [], "tokens": [],
            })
            entry["n_prompts"] += 1
            entry["peaks"].append(f["max_activation"] or 0.0)
            entry["tokens"].append(f["max_token"])

    ranked = sorted(
        by_feature.values(),
        key=lambda e: (e["n_prompts"], sum(e["peaks"]) / len(e["peaks"])),
        reverse=True,
    )[:top_k]
    features = [{
        "source":      e["source"],
        "index":       e["index"],
        "explanation": e["explanation"],
        "fires_on":    f"{e['n_prompts']}/{len(prompts)} prompts",
        "mean_peak":   round(sum(e["peaks"]) / len(e["peaks"]), 3),
        "peak_tokens": sorted({t for t in e["tokens"] if t}),
        "url":         e["url"],
    } for e in ranked]

    raw_path = output_dir / "feature_search.json"
    raw_path.write_text(json.dumps(raw, indent=1))
    return InterpResult(
        plot_paths=[],
        data={"model_id": model_id, "layers": layers, "features": features, "raw_path": str(raw_path)},
    )
