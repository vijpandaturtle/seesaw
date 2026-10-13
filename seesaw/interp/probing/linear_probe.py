"""Linear probe: is a concept linearly decodable from the residual stream, and from which layer?

Two prompt sets that differ in the concept (female- vs male-stereotyped
occupations, say). At every layer, take the residual stream at each prompt's
final token and fit a logistic-regression probe; cross-validated accuracy
says how linearly separable the two sets are there.

It also returns the difference-of-means direction at the best layer — the
mean activation of set A minus set B. That vector is what steering and
directional ablation take as input, which is how a probe turns into an
intervention.

Decodable is not the same as used: a probe can find a concept the model never
reads. Pair this with directional ablation before claiming the model relies on it.

Answers: "Where in the model is this concept represented?"
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from transformer_lens import HookedTransformer

matplotlib.use("Agg")

from ..metrics import last_positions
from ..result import InterpResult


def _final_resids(model, prompts) -> torch.Tensor:
    """[n_layers, n_prompts, d_model] residual stream after each block, at each prompt's last token."""
    tokens = model.to_tokens(prompts)
    pos = last_positions(model, prompts).to(tokens.device)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n.endswith("hook_resid_post"))
    rows = torch.arange(len(prompts), device=tokens.device)
    return torch.stack([
        cache["resid_post", layer][rows, pos].float().cpu() for layer in range(model.cfg.n_layers)
    ])


def direction(model: HookedTransformer, prompts_a: list[str], prompts_b: list[str], layer: int) -> list[float]:
    """Difference of means, mean(A) - mean(B), of the residual stream after `layer` at each prompt's last token."""
    a = _final_resids(model, prompts_a)[layer]
    b = _final_resids(model, prompts_b)[layer]
    return (a.mean(0) - b.mean(0)).tolist()


def _fit_predict(x_train, y_train, x_test, weight_decay: float, steps: int = 200) -> torch.Tensor:
    """L2-regularised logistic regression, fit with LBFGS on standardised features."""
    mu, sd = x_train.mean(0), x_train.std(0) + 1e-6
    x_train, x_test = (x_train - mu) / sd, (x_test - mu) / sd
    w = torch.zeros(x_train.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x_train @ w + b, y_train)
        loss = loss + weight_decay * (w ** 2).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    with torch.no_grad():
        return (x_test @ w + b) > 0


def compute(
    model: HookedTransformer,
    prompts_a: list[str],
    prompts_b: list[str],
    output_dir: Path,
    n_folds: int = 5,
    weight_decay: float = 1e-2,
    seed: int = 0,
) -> InterpResult:
    """Fit a probe separating prompts_a from prompts_b at every layer.

    Args:
        model: Loaded HookedTransformer.
        prompts_a: Prompts with the concept (label 1).
        prompts_b: Prompts without it, or with its opposite (label 0).
        output_dir: Directory to save the plot.
        n_folds: Cross-validation folds (capped at the smaller set's size).
        weight_decay: L2 penalty; small datasets in high dimensions overfit without it.
        seed: Seed for the fold split.

    Returns:
        InterpResult with per-layer accuracy, the best layer, and the
        difference-of-means direction there.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    acts = torch.cat([_final_resids(model, prompts_a), _final_resids(model, prompts_b)], dim=1)
    labels = torch.tensor([1.0] * len(prompts_a) + [0.0] * len(prompts_b))
    n = len(labels)
    n_folds = max(2, min(n_folds, len(prompts_a), len(prompts_b)))
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    folds = [order[i::n_folds] for i in range(n_folds)]

    accuracies = []
    for layer_acts in acts:
        correct = 0
        for test_idx in folds:
            train_mask = torch.ones(n, dtype=torch.bool)
            train_mask[test_idx] = False
            pred = _fit_predict(layer_acts[train_mask], labels[train_mask], layer_acts[test_idx], weight_decay)
            correct += (pred.float() == labels[test_idx]).sum().item()
        accuracies.append(correct / n)

    # Earliest layer within a hair of the best: later layers often tie, and
    # the earliest is where the concept first becomes available.
    best_acc = max(accuracies)
    best_layer = next(i for i, a in enumerate(accuracies) if a >= best_acc - 1e-9)
    direction = acts[best_layer, : len(prompts_a)].mean(0) - acts[best_layer, len(prompts_a):].mean(0)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(accuracies)), accuracies, marker="o", color="#d73027")
    ax.axhline(0.5, color="grey", linestyle="--", label="chance")
    ax.axvline(best_layer, color="black", linewidth=0.8)
    ax.set_xlabel("Layer (resid_post)")
    ax.set_ylabel(f"{n_folds}-fold CV accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Linear Probe — best layer {best_layer} ({best_acc:.0%})")
    ax.legend()
    plt.tight_layout()
    plot_path = output_dir / "linear_probe.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return InterpResult(
        plot_paths=[plot_path],
        data={
            "accuracies":     [round(a, 4) for a in accuracies],
            "best_layer":     best_layer,
            "best_accuracy":  round(best_acc, 4),
            "direction":      direction.tolist(),
            "direction_norm": round(direction.norm().item(), 4),
            "n_per_class":    [len(prompts_a), len(prompts_b)],
        },
    )
