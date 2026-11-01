"""A small client for the Neuronpedia API — feature labels, top logits, and which features fire on a text.

Neuronpedia hosts SAEs for many open models along with an auto-generated
explanation for each feature. Seesaw uses it for three things:

- `get_feature` — the human-readable side of a feature found locally: its
  explanation, the tokens it promotes and suppresses, and a dashboard link.
- `search_text` — which features fire most on a piece of text, computed on
  Neuronpedia's servers. No SAE download, no local model: the cheapest way to
  see what features a prompt involves, and it works for models too large to
  run here.
- `get_graph` / `generate_graph` — attribution graphs (circuit tracing, Ameisen
  et al. 2025) computed by circuit-tracer on Neuronpedia's GPUs. Reading an
  existing graph is free and private; generating one PUBLISHES it: the graph
  and its prompt are stored on Neuronpedia under a public URL.

Responses are cached on disk (`~/.cache/seesaw/neuronpedia`, or
`SEESAW_NEURONPEDIA_CACHE`). Neuronpedia's explanations can be regenerated
and its servers can be down; a cache keeps a finished analysis reproducible
and lets evals rerun without the network.

`NEURONPEDIA_API_KEY` is sent if set; the read endpoints used here don't need one.
Standard library only, so no new dependency.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://www.neuronpedia.org"
TIMEOUT_S = 30


class NeuronpediaError(RuntimeError):
    """Neuronpedia couldn't be reached or returned something unusable."""


def _cache_dir() -> Path:
    return Path(os.environ.get("SEESAW_NEURONPEDIA_CACHE", Path.home() / ".cache/seesaw/neuronpedia"))


def _request(method: str, path: str, body: dict | None = None, use_cache: bool = True,
             timeout: float = TIMEOUT_S) -> dict:
    key = hashlib.sha256(json.dumps([method, path, body], sort_keys=True).encode()).hexdigest()[:24]
    cached = _cache_dir() / f"{key}.json"
    if use_cache and cached.exists():
        return json.loads(cached.read_text())

    headers = {"Content-Type": "application/json", "User-Agent": "seesaw"}
    if os.environ.get("NEURONPEDIA_API_KEY"):
        headers["x-api-key"] = os.environ["NEURONPEDIA_API_KEY"]
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE_URL + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        raise NeuronpediaError(f"{method} {path} failed: HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise NeuronpediaError(f"{method} {path} failed: {exc}") from exc

    if use_cache:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(payload))
    return payload


def feature_url(source: str, index: int) -> str:
    """Dashboard link, e.g. feature_url('gpt2-small/8-res-jb', 1000)."""
    return f"{BASE_URL}/{source}/{index}"


def get_feature(source: str, index: int, use_cache: bool = True) -> dict:
    """One feature's explanation and logit effects.

    Args:
        source: '<model>/<sae>' as Neuronpedia names it, e.g. 'gpt2-small/8-res-jb'
            (`sae.neuronpedia_source(sae)` gives it for a SAELens SAE).
        index: Feature index within the SAE.

    Returns:
        {"source", "index", "explanation", "promotes", "suppresses",
         "max_activation", "url"} — explanation is None if none exists yet.
    """
    raw = _request("GET", f"/api/feature/{source}/{index}", use_cache=use_cache)
    explanations = raw.get("explanations") or []
    return {
        "source":         source,
        "index":          int(index),
        "explanation":    explanations[0]["description"].strip() if explanations else None,
        "promotes":       (raw.get("pos_str") or [])[:10],
        "suppresses":     (raw.get("neg_str") or [])[:10],
        "max_activation": raw.get("maxActApprox"),
        "url":            feature_url(source, index),
    }


def search_text(
    model_id: str,
    source_set: str,
    layers: list[str],
    text: str,
    n: int = 20,
    use_cache: bool = True,
) -> dict:
    """The features that fire most strongly anywhere in `text`, computed by Neuronpedia.

    Args:
        model_id: Neuronpedia model id, e.g. 'gpt2-small'.
        source_set: SAE family, e.g. 'res-jb'.
        layers: SAE ids to search, e.g. ['8-res-jb'].
        text: The text to run.
        n: How many features to return.

    Returns:
        {"tokens": [...], "features": [{"source", "index", "max_activation",
         "max_token", "activations", "explanation", "url"}, ...]}, strongest first.
    """
    raw = _request("POST", "/api/search-all", {
        "modelId": model_id,
        "sourceSet": source_set,
        "text": text,
        "selectedLayers": layers,
        "sortIndexes": [],
        "ignoreBos": True,
        "densityThreshold": -1,
        "numResults": n,
    }, use_cache=use_cache)
    tokens = raw.get("tokens") or []
    features = []
    for hit in raw.get("result") or []:
        source = f"{hit['modelId']}/{hit['layer']}"
        explanations = (hit.get("neuron") or {}).get("explanations") or []
        idx = hit.get("maxValueIndex")
        features.append({
            "source":         source,
            "index":          int(hit["index"]),
            "max_activation": hit.get("maxValue"),
            "max_token":      tokens[idx] if idx is not None and idx < len(tokens) else None,
            "activations":    hit.get("values"),
            "explanation":    explanations[0]["description"].strip() if explanations else None,
            "url":            feature_url(source, hit["index"]),
        })
    return {"tokens": tokens, "features": features}


# ── attribution graphs ───────────────────────────────────────────────────────
GRAPH_TIMEOUT_S = 300           # generation runs circuit-tracer on a GPU; it takes a while


def _download_json(url: str, use_cache: bool = True) -> dict:
    key = hashlib.sha256(url.encode()).hexdigest()[:24]
    cached = _cache_dir() / "graphs" / f"{key}.json"
    if use_cache and cached.exists():
        return json.loads(cached.read_text())
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "seesaw"}),
                                    timeout=GRAPH_TIMEOUT_S) as resp:
            payload = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise NeuronpediaError(f"downloading graph {url} failed: {exc}") from exc
    if use_cache:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(payload))
    return payload


def graph_url(model_id: str, slug: str) -> str:
    """The graph's page in Neuronpedia's attribution-graph viewer."""
    return f"{BASE_URL}/{model_id}/graph?slug={slug}"


def get_graph(model_id: str, slug: str, use_cache: bool = True) -> dict:
    """Fetch an existing attribution graph by slug. Read-only.

    Returns:
        The graph JSON: {"metadata", "nodes", "links", ...} in the Anthropic /
        circuit-tracer format, plus "source_set" (the transcoder set whose
        Neuronpedia dashboards label its features).

    Raises:
        NeuronpediaError: no graph with that slug, or Neuronpedia unreachable.
    """
    meta = _request("GET", f"/api/graph/{model_id}/{slug}", use_cache=use_cache)
    if "url" not in meta:
        raise NeuronpediaError(f"no graph {model_id}/{slug}: {meta.get('error', meta)}")
    graph = _download_json(meta["url"], use_cache=use_cache)
    graph["source_set"] = meta.get("sourceSetName")
    return graph


def generate_graph(
    prompt: str,
    slug: str,
    model_id: str = "gemma-2-2b",
    max_feature_nodes: int = 5000,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.85,
) -> dict:
    """Have Neuronpedia run circuit-tracer on `prompt`. PUBLISHES the graph.

    The graph, its prompt and the slug are saved on Neuronpedia and anyone with
    the URL can see them — don't send a prompt you wouldn't post. If a graph
    with this slug already exists it is returned instead of regenerated.

    Args:
        prompt: Text to trace (Neuronpedia caps it at 64 tokens).
        slug: Unique id for the graph: lowercase letters, digits, - and _.
        model_id: A model Neuronpedia generates graphs for (gemma-2-2b).
        max_feature_nodes / node_threshold / edge_threshold: circuit-tracer's
            pruning settings; Neuronpedia's defaults.

    Returns:
        The graph JSON, as get_graph returns it.
    """
    try:
        return get_graph(model_id, slug)
    except NeuronpediaError:
        pass
    _request("POST", "/api/graph/generate", {
        "prompt": prompt, "modelId": model_id, "slug": slug,
        "maxFeatureNodes": max_feature_nodes, "nodeThreshold": node_threshold,
        "edgeThreshold": edge_threshold,
    }, use_cache=False, timeout=GRAPH_TIMEOUT_S)
    return get_graph(model_id, slug)

