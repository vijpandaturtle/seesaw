"""Circuit evaluation: how much of the behaviour does a set of heads account for?

Two complementary scores, both from mean-ablating heads (Wang et al. 2022):

- faithfulness — ablate every head *outside* the circuit. If the circuit is
  the mechanism, the behaviour survives: logit diff stays near the full model's.
- necessity — ablate the circuit itself. If the circuit is the mechanism, the
  behaviour collapses.

Each score is compared against random circuits of the same size. A circuit
that keeps 60% of the logit diff sounds good until random sets of the same
size keep 55%; the gap to the random baseline is the real evidence.

Means are taken per position over `reference_prompts`. For IOI the right
reference is the ABC distribution (three different names), which removes
name-specific information while keeping the sentence structure; the default,
the prompts themselves, is a weaker ablation.

Answers: "Is this set of heads actually the circuit, or just part of it?"
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from transformer_lens import HookedTransformer

matplotlib.use("Agg")

from .._helpers import head_name, parse_head, tokens_to_ids
from ..metrics import logit_diff
from ..result import InterpResult


def _ld_with_ablated(model, tokens, ablate: set[tuple[int, int]], means, pos_ids, neg_ids) -> float:
    by_layer: dict[int, list[int]] = {}
    for layer, head in ablate:
        by_layer.setdefault(layer, []).append(head)

    def make_hook(hs):
        def hook_fn(value, hook):
            value = value.clone()
            value[:, :, hs, :] = means[hook.name][:, :, hs, :]
            return value
        return hook_fn

    hooks = [(f"blocks.{l}.attn.hook_z", make_hook(hs)) for l, hs in by_layer.items()]
    with torch.no_grad(), model.hooks(fwd_hooks=hooks):
        logits = model(tokens)
    return logit_diff(logits, pos_ids, neg_ids).mean().item()


def compute(
    model: HookedTransformer,
    prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    circuit: list[str],
    output_dir: Path,
    reference_prompts: list[str] | None = None,
    n_random: int = 10,
    seed: int = 0,
) -> InterpResult:
    """Score a candidate circuit's faithfulness and necessity against random circuits.

    Args:
        model: Loaded HookedTransformer.
        prompts: Prompts the behaviour is measured on.
        positive_tokens: Correct answer token per prompt.
        negative_tokens: Distractor token per prompt.
        circuit: Heads in the candidate circuit, e.g. ["L9H9", "L9H6", "L7H3"].
        output_dir: Directory to save the plot.
        reference_prompts: Prompts the ablation means come from; must tokenize
            to the same length as `prompts`. Defaults to `prompts`.
        n_random: Random same-size circuits to compare against.
        seed: Seed for drawing the random circuits.

    Returns:
        InterpResult with faithfulness, necessity and their random baselines.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens = model.to_tokens(prompts)
    ref_tokens = model.to_tokens(reference_prompts) if reference_prompts else tokens
    if ref_tokens.shape[1] != tokens.shape[1]:
        raise ValueError("reference_prompts must tokenize to the same length as prompts")
    pos_ids = tokens_to_ids(model, positive_tokens)
    neg_ids = tokens_to_ids(model, negative_tokens)
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    all_heads = {(l, h) for l in range(n_layers) for h in range(n_heads)}
    members = {parse_head(h) for h in circuit}

    with torch.no_grad():
        logits = model(tokens)
        _, cache = model.run_with_cache(ref_tokens, names_filter=lambda n: n.endswith("hook_z"))
    full_ld = logit_diff(logits, pos_ids, neg_ids).mean().item()
    # Per-position mean over the reference batch: [1, seq, n_heads, d_head].
    means = {name: cache[name].mean(0, keepdim=True) for name in cache.keys()}
    del cache

    def scores(heads: set) -> tuple[float, float]:
        kept = _ld_with_ablated(model, tokens, all_heads - heads, means, pos_ids, neg_ids)
        removed = _ld_with_ablated(model, tokens, heads, means, pos_ids, neg_ids)
        faith = kept / full_ld if abs(full_ld) > 1e-6 else 0.0
        need = 1 - removed / full_ld if abs(full_ld) > 1e-6 else 0.0
        return faith, need

    faithfulness, necessity = scores(members)

    gen = torch.Generator().manual_seed(seed)
    pool = sorted(all_heads)
    random_faith, random_need = [], []
    for _ in range(n_random):
        idx = torch.randperm(len(pool), generator=gen)[: len(members)]
        f, n = scores({pool[i] for i in idx.tolist()})
        random_faith.append(f)
        random_need.append(n)
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, title, ours, rand in [
        (axes[0], "Faithfulness (only circuit kept)", faithfulness, random_faith),
        (axes[1], "Necessity (circuit removed)", necessity, random_need),
    ]:
        ax.scatter([0] * len(rand), rand, color="grey", alpha=0.6, label="random circuits")
        ax.scatter([1], [ours], color="#d73027", s=80, label="candidate circuit")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["random", "candidate"])
        ax.set_xlim(-0.5, 1.5)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(1, color="grey", linewidth=0.8, linestyle="--")
        ax.set_title(title)
    axes[0].legend(loc="upper left", fontsize=8)
    plt.suptitle(f"Circuit Evaluation — {len(members)} heads")
    plt.tight_layout()
    plot_path = output_dir / "circuit_evaluation.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return InterpResult(
        plot_paths=[plot_path],
        data={
            "circuit":              sorted(head_name(l, h) for l, h in members),
            "full_ld":              round(full_ld, 4),
            "faithfulness":         round(faithfulness, 4),
            "necessity":            round(necessity, 4),
            "random_faithfulness":  round(mean(random_faith), 4),
            "random_necessity":     round(mean(random_need), 4),
            "n_random":             n_random,
        },
    )
