"""Circuit tracing: the attribution graph behind one prediction, from Neuronpedia.

An attribution graph (Ameisen et al. 2025, "Circuit Tracing") replaces the
model's MLPs with transcoders, freezes attention, and records how much each
transcoder feature, input token and error term contributes linearly to every
later one, down to the predicted token. Neuronpedia runs Anthropic's
circuit-tracer on Gemma-2-2B and serves the graphs, so nothing runs locally.

The raw graph has hundreds of nodes and tens of thousands of edges. This tool
reduces it to what can be read:
- influence on the target token, summed over every path (not only direct edges),
  computed the way circuit-tracer prunes: each node's incoming edge weights are
  normalised, then scores flow backwards from the target;
- the top features by that influence, labelled from Neuronpedia's dashboards;
- how much influence passes through error nodes — the part of the model the
  transcoders don't explain. A high share means the readable circuit is
  incomplete, however clean it looks.

An estimate, not an intervention: edges are linear attributions through a
replacement model. Checking a feature's role needs steering or ablation.

Generating a new graph PUBLISHES it on Neuronpedia (see neuronpedia.generate_graph);
passing the slug of an existing graph only reads.

Answers: "Which features, through which intermediate steps, produce this token?"
"""

from __future__ import annotations

import hashlib
from collections import defaultdict, deque
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

from ..feature_extraction import neuronpedia
from ..result import InterpResult

FEATURE_TYPES = {"cross layer transcoder", "transcoder", "mlp transcoder"}
LAYER_STRIDE = 100_000      # graph feature ids are layer * 100000 + index


def default_slug(prompt: str, model_id: str) -> str:
    return "seesaw-" + hashlib.sha256(f"{model_id}|{prompt}".encode()).hexdigest()[:12]


def _influence(nodes: dict, links: list[dict], target: str) -> dict[str, float]:
    """Each node's total influence on `target`, over all paths.

    Incoming weights of every node are normalised to sum to 1 in absolute
    value, then influence flows backwards: I(target) = 1 and
    I(u) = Σ_v |w(u→v)| / Σ_x |w(x→v)| · I(v).
    """
    incoming_total: dict[str, float] = defaultdict(float)
    out_edges: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for link in links:
        incoming_total[link["target"]] += abs(link["weight"])
        out_edges[link["source"]].append((link["target"], abs(link["weight"])))

    in_edges: dict[str, list[str]] = defaultdict(list)
    for link in links:
        in_edges[link["target"]].append(link["source"])

    # Reverse topological order: a node is scored once every node it feeds is.
    score: dict[str, float] = {n: 0.0 for n in nodes}
    score[target] = 1.0
    order = []
    pending = {n: len(out_edges[n]) for n in nodes}
    ready = deque(n for n, k in pending.items() if k == 0)
    while ready:
        v = ready.popleft()
        order.append(v)
        for u in in_edges[v]:
            pending[u] -= 1
            if pending[u] == 0:
                ready.append(u)
    for u in order:
        if u == target:
            continue
        score[u] = sum(
            w / incoming_total[v] * score[v] for v, w in out_edges[u] if incoming_total[v] > 0
        )
    return score


def _label(node: dict, model_id: str, source_set: str | None, explain: bool) -> tuple[str | None, str | None]:
    """(explanation, dashboard url) for a transcoder feature node."""
    if node.get("clerp"):
        return node["clerp"], None
    if not source_set:
        return None, None
    layer = int(node["layer"])
    index = node["feature"] - layer * LAYER_STRIDE
    index = index if 0 <= index < LAYER_STRIDE else node["feature"]
    source = f"{model_id}/{layer}-{source_set}"
    if not explain:
        return None, neuronpedia.feature_url(source, index)
    try:
        f = neuronpedia.get_feature(source, index)
        return f["explanation"], f["url"]
    except neuronpedia.NeuronpediaError:
        return None, neuronpedia.feature_url(source, index)


def compute(
    output_dir: Path,
    prompt: str | None = None,
    slug: str | None = None,
    model_id: str = "gemma-2-2b",
    allow_generate: bool = False,
    top_k: int = 15,
    explain: bool = True,
) -> InterpResult:
    """Fetch (or generate) an attribution graph and summarise it.

    Args:
        output_dir: Where the plot and the graph JSON are written.
        prompt: Text to trace. Needed only when generating.
        slug: An existing graph's slug, e.g. 'gemma-fact-dallas-austin'. If
            omitted, one is derived from the prompt, so the same prompt maps to
            the same graph and is only generated once.
        model_id: Neuronpedia model id; graph generation supports gemma-2-2b.
        allow_generate: Generate the graph if the slug doesn't exist yet. This
            publishes the prompt and graph on Neuronpedia; it is off by default.
        top_k: Features to report.
        explain: Look up feature explanations on Neuronpedia.

    Returns:
        InterpResult with the target token, the top features by influence, the
        strongest edges among them, the error-node share and the graph's URL.
    """
    import json

    if slug is None and prompt is None:
        raise ValueError("give a slug of an existing graph, or a prompt")
    slug = slug or default_slug(prompt, model_id)
    try:
        graph = neuronpedia.get_graph(model_id, slug)
    except neuronpedia.NeuronpediaError:
        if not (allow_generate and prompt):
            raise neuronpedia.NeuronpediaError(
                f"no graph {model_id}/{slug} on Neuronpedia. Generating one publishes the prompt; "
                f"pass allow_generate=True to do that.")
        graph = neuronpedia.generate_graph(prompt, slug, model_id=model_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"graph_{slug}.json").write_text(json.dumps(graph))
    nodes = {n["node_id"]: n for n in graph["nodes"]}
    links = [l for l in graph["links"] if l["source"] in nodes and l["target"] in nodes]
    logits = [n for n in graph["nodes"] if n["feature_type"] == "logit"]
    target = next((n for n in logits if n.get("is_target_logit")), None) or max(
        logits, key=lambda n: n.get("token_prob") or 0)
    tokens = graph["metadata"].get("prompt_tokens") or []

    score = _influence(nodes, links, target["node_id"])
    features = [n for n in graph["nodes"] if n["feature_type"] in FEATURE_TYPES]
    errors = [n for n in graph["nodes"] if "error" in n["feature_type"]]
    embeddings = [n for n in graph["nodes"] if n["feature_type"] == "embedding"]
    mass = lambda ns: sum(score[n["node_id"]] for n in ns)  # noqa: E731
    total = mass(features) + mass(errors) + mass(embeddings)
    error_share = mass(errors) / total if total > 0 else 0.0

    ranked = sorted(features, key=lambda n: -score[n["node_id"]])[:top_k]
    top = []
    for n in ranked:
        label, url = _label(n, model_id, graph.get("source_set"), explain)
        pos = n["ctx_idx"]
        top.append({
            "node_id":   n["node_id"],
            "layer":     int(n["layer"]),
            "position":  pos,
            "token":     tokens[pos] if pos < len(tokens) else None,
            "influence": round(score[n["node_id"]], 4),
            "explanation": label,
            "url":       url,
        })

    keep = {f["node_id"] for f in top} | {target["node_id"]}
    edges = sorted(
        (l for l in links if l["source"] in keep and l["target"] in keep),
        key=lambda l: -abs(l["weight"]),
    )[: 2 * top_k]
    name = {f["node_id"]: f"L{f['layer']}@{f['token']!r}: {(f['explanation'] or '?')[:30]}" for f in top}
    target_label = target.get("clerp") or f"logit {target['feature']}"
    name[target["node_id"]] = target_label
    top_edges = [
        {"source": name[l["source"]], "target": name[l["target"]], "weight": round(l["weight"], 4)}
        for l in edges
    ]

    # ── Plot: one column per token position that has features, layer up ─────
    used = sorted({f["position"] for f in top} | {target["ctx_idx"]})
    col = {p: 4 * i for i, p in enumerate(used)}
    n_layers = max([int(n["layer"]) for n in logits] + [1])
    cell_size = defaultdict(int)
    for f in top:
        cell_size[(f["position"], f["layer"])] += 1
    xy, side, stack = {}, {}, defaultdict(int)
    for f in sorted(top, key=lambda f: (f["position"], f["layer"])):
        cell = (f["position"], f["layer"])
        k = stack[cell]                                  # nodes sharing a cell sit side by side
        stack[cell] += 1
        xy[f["node_id"]] = (col[f["position"]] + 0.8 * k, f["layer"])
        if cell_size[cell] > 1:                          # first label left, the rest right
            side[f["node_id"]] = -1 if k == 0 else 1
        else:                                            # lone nodes alternate to spread labels
            side[f["node_id"]] = 1 if len(side) % 2 == 0 else -1
    xy[target["node_id"]] = (col[target["ctx_idx"]], n_layers + 1)

    fig, ax = plt.subplots(figsize=(max(8, 3.2 * len(used) + 4), 9))
    wmax = max([abs(l["weight"]) for l in edges] + [1e-9])
    for l in edges:
        (x0, y0), (x1, y1) = xy[l["source"]], xy[l["target"]]
        ax.plot([x0, x1], [y0, y1], color="#d73027" if l["weight"] > 0 else "#4575b4",
                alpha=0.2 + 0.6 * abs(l["weight"]) / wmax, linewidth=0.5 + 3 * abs(l["weight"]) / wmax,
                zorder=1)
    imax = max([f["influence"] for f in top] + [1e-9])
    for f in top:
        x, y = xy[f["node_id"]]
        ax.scatter([x], [y], s=60 + 400 * f["influence"] / imax, color="#fdae61",
                   edgecolor="black", zorder=3)
        right = side[f["node_id"]] > 0
        ax.annotate((f["explanation"] or f"#{f['node_id']}")[:34], (x, y), textcoords="offset points",
                    xytext=(9 if right else -9, 0), ha="left" if right else "right", va="center",
                    fontsize=7, zorder=4,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    tx, ty = xy[target["node_id"]]
    ax.scatter([tx], [ty], s=300, marker="s", color="#1a9850", zorder=3)
    ax.annotate(target_label, (tx, ty), textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks([col[p] for p in used])
    ax.set_xticklabels([repr(tokens[p]) if p < len(tokens) else str(p) for p in used], fontsize=9)
    ax.set_xlim(-4, col[used[-1]] + 4)
    ax.set_ylabel("Layer")
    ax.set_ylim(-1, n_layers + 3)
    ax.set_title(f"Circuit Tracing — {graph['metadata'].get('scan', model_id)}: top {len(top)} features "
                 f"by influence on the output\n{error_share:.0%} of influence passes through error nodes "
                 f"(not explained by the transcoders)")
    plt.tight_layout()
    plot_path = output_dir / f"circuit_tracing_{slug}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return InterpResult(
        plot_paths=[plot_path],
        data={
            "slug":          slug,
            "graph_url":     neuronpedia.graph_url(model_id, slug),
            "prompt":        graph["metadata"].get("prompt"),
            "target":        target_label,
            "target_prob":   target.get("token_prob"),
            "n_nodes":       len(nodes),
            "n_links":       len(links),
            "error_influence_share": round(error_share, 4),
            "top_features":  top,
            "top_edges":     top_edges,
        },
    )
