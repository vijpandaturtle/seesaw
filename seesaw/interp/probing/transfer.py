"""Probe transfer: does a probe trained on one set of texts still work on different ones?

The question a probe deployed as a monitor has to answer. A probe that
scores well under cross-validation can still be reading something incidental
to the training texts (their topic, their length, one word they share) and
fail on the texts it will actually see. So it's judged on a held-out set,
ideally from a different distribution, against three baselines:

- shuffled labels: what a probe at the chosen layer scores on nothing;
- layer 0: a probe on the embeddings alone, i.e. on which words appear. If it
  does as well, the model's processing adds nothing;
- a bag-of-words classifier on the raw tokens, with no model at all: the
  cheapest monitor anyone could deploy instead.

The layer is chosen by cross-validation on the training set only, so the
test set never influences it.

Answers: "Would a probe on this model's activations make a good detector for X?"
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from transformer_lens import HookedTransformer

matplotlib.use("Agg")

from ..result import InterpResult
from .linear_probe import _final_resids, _fit_predict


def _accuracy(x_train, y_train, x_test, y_test, weight_decay) -> float:
    pred = _fit_predict(x_train, y_train, x_test, weight_decay)
    return (pred.float() == y_test).float().mean().item()


def _cv(x, y, n_folds, weight_decay, gen) -> float:
    n = len(y)
    order = torch.randperm(n, generator=gen)
    correct = 0
    for i in range(n_folds):
        test = order[i::n_folds]
        mask = torch.ones(n, dtype=torch.bool)
        mask[test] = False
        correct += _accuracy(x[mask], y[mask], x[test], y[test], weight_decay) * len(test)
    return correct / n


def _bag_of_words(model, train: list[str], test: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-count vectors over the training vocabulary, for the no-model baseline."""
    def ids(texts):
        return [model.to_tokens(t, prepend_bos=False)[0].tolist() for t in texts]
    train_ids, test_ids = ids(train), ids(test)
    vocab = {t: i for i, t in enumerate(sorted({t for row in train_ids for t in row}))}

    def counts(rows):
        out = torch.zeros(len(rows), max(1, len(vocab)))
        for r, row in enumerate(rows):
            for t in row:
                if t in vocab:
                    out[r, vocab[t]] += 1
        return out
    return counts(train_ids), counts(test_ids)


def compute(
    model: HookedTransformer,
    prompts: list[str],
    contrast_prompts: list[str],
    test_prompts: list[str],
    test_contrast_prompts: list[str],
    output_dir: Path,
    n_folds: int = 5,
    weight_decay: float = 1e-2,
    n_shuffles: int = 5,
    seed: int = 0,
) -> InterpResult:
    """Choose a layer on the training texts, then score it and the baselines on the test texts.

    Args:
        model: Loaded HookedTransformer.
        prompts: Training texts with the property (label 1).
        contrast_prompts: Training texts without it (label 0).
        test_prompts: Held-out texts with the property.
        test_contrast_prompts: Held-out texts without it.
        output_dir: Directory to save the plot.
        n_folds: Cross-validation folds on the training texts.
        weight_decay: L2 penalty for every probe.
        n_shuffles: Shuffled-label refits at the chosen layer.
        seed: Seed for folds and shuffles.

    Returns:
        InterpResult with per-layer train CV and test accuracy, the chosen
        layer, and the shuffled, layer-0 and bag-of-words baselines on the test set.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    train = torch.cat([_final_resids(model, prompts), _final_resids(model, contrast_prompts)], dim=1)
    test = torch.cat([_final_resids(model, test_prompts), _final_resids(model, test_contrast_prompts)], dim=1)
    with torch.no_grad():
        embeds = [model.embed(model.to_tokens(t))[0].mean(0).float().cpu()
                  for t in prompts + contrast_prompts + test_prompts + test_contrast_prompts]
    y_train = torch.tensor([1.0] * len(prompts) + [0.0] * len(contrast_prompts))
    y_test = torch.tensor([1.0] * len(test_prompts) + [0.0] * len(test_contrast_prompts))
    n_folds = max(2, min(n_folds, len(prompts), len(contrast_prompts)))

    cv = [_cv(layer, y_train, n_folds, weight_decay, torch.Generator().manual_seed(seed)) for layer in train]
    test_acc = [_accuracy(tr, y_train, te, y_test, weight_decay) for tr, te in zip(train, test)]
    best = max(range(len(cv)), key=lambda l: (cv[l], -l))

    gen = torch.Generator().manual_seed(seed + 1)
    shuffled = [_accuracy(train[best], y_train[torch.randperm(len(y_train), generator=gen)],
                          test[best], y_test, weight_decay) for _ in range(n_shuffles)]
    n_train = len(y_train)
    embed_train, embed_test = torch.stack(embeds[:n_train]), torch.stack(embeds[n_train:])
    layer0 = _accuracy(embed_train, y_train, embed_test, y_test, weight_decay)
    bow_train, bow_test = _bag_of_words(
        model, prompts + contrast_prompts, test_prompts + test_contrast_prompts)
    text_baseline = _accuracy(bow_train, y_train, bow_test, y_test, weight_decay)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(cv, marker="o", color="#4575b4", label="train (cross-validated)")
    ax.plot(test_acc, marker="o", color="#d73027", label="held-out test")
    ax.axhline(text_baseline, color="grey", linestyle="--", label="bag-of-words (no model)")
    ax.axhline(layer0, color="black", linestyle=":", label="embeddings only")
    ax.axvline(best, color="black", linewidth=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Layer (resid_post)")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Probe transfer — layer {best}: {test_acc[best]:.0%} on held-out texts")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plot_path = output_dir / "probe_transfer.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return InterpResult(
        plot_paths=[plot_path],
        data={
            "train_accuracies":  [round(a, 4) for a in cv],
            "test_accuracies":   [round(a, 4) for a in test_acc],
            "best_layer":        best,
            "train_accuracy":    round(cv[best], 4),
            "test_accuracy":     round(test_acc[best], 4),
            "shuffled_accuracy": round(sum(shuffled) / len(shuffled), 4) if shuffled else None,
            "embedding_accuracy": round(layer0, 4),
            "text_baseline_accuracy": round(text_baseline, 4),
            "n_train":           [len(prompts), len(contrast_prompts)],
            "n_test":            [len(test_prompts), len(test_contrast_prompts)],
        },
    )
