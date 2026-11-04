"""Probe monitor: would a probe on this model's activations make a good detector for X?

The practitioner's version of probing: not "is X represented?" but "can I
deploy a cheap classifier on the activations to flag X at runtime, and is it
better than not using the model's internals at all?" (hallucination,
toxicity, prompt injection, a topic you need to route).

Train → choose → transfer → compare:

1. Train — fit a probe at every layer on the training texts, choosing the
   layer by cross-validation there alone.
2. Transfer — score that layer's probe on held-out texts. Give them from a
   different distribution (other topics, other phrasing, other sources) if
   the monitor will see different texts in use: that's the case that fails.
3. Compare — against shuffled labels (noise), a probe on the embeddings
   alone (which words appear), and a bag-of-words classifier with no model.

Verdicts:
- no_signal          — the probe does no better than shuffled labels on the training texts.
- does_not_transfer  — it works on the training texts but not the held-out ones.
- lexical_only       — it transfers, but the words alone do as well: use the
                       cheaper classifier.
- monitor_works      — it transfers and beats both text-only baselines.
"""

from __future__ import annotations

from pathlib import Path

from transformer_lens import HookedTransformer

from seesaw.interp.probing import linear_probe, transfer

from .result import PathwayResult

# Training CV must beat shuffled labels by this much to count as signal.
SIGNAL_MARGIN = 0.2
# Held-out accuracy needed to call it a working detector.
TRANSFER_MIN = 0.75
# The probe must beat the better text-only baseline by this much on held-out texts.
OVER_TEXT_MARGIN = 0.1


def run(
    model: HookedTransformer,
    prompts: list[str],
    contrast_prompts: list[str],
    test_prompts: list[str],
    test_contrast_prompts: list[str],
    output_dir: Path,
) -> PathwayResult:
    """Run the probe-monitor pathway.

    Args:
        model: Loaded HookedTransformer.
        prompts: Training texts with the property to detect.
        contrast_prompts: Training texts without it.
        test_prompts: Held-out texts with it, ideally from a different distribution.
        test_contrast_prompts: Held-out texts without it.
        output_dir: Where every step saves its plots.

    Returns:
        PathwayResult with the chosen layer, held-out accuracy, the three
        baselines and a verdict.
    """
    result = PathwayResult(pathway="probe_monitor")

    # ── 1. Train ──────────────────────────────────────────────────────────────
    probe = linear_probe.compute(model, prompts, contrast_prompts, output_dir)
    result.steps["linear_probe"] = probe
    train_acc, shuffled = probe.data["best_accuracy"], probe.data["shuffled_accuracy"]
    result.findings.update({"train_accuracy": train_acc, "shuffled_accuracy": shuffled})
    result.evidence.append(
        f"On the training texts, the best layer's probe scores {train_acc:.0%} under cross-validation; "
        f"on shuffled labels, {shuffled:.0%}.")
    if train_acc - shuffled < SIGNAL_MARGIN:
        return result.stop("no_signal", f"the probe beats shuffled labels by under {SIGNAL_MARGIN:.0%}")

    # ── 2-3. Transfer and compare ─────────────────────────────────────────────
    tr = transfer.compute(model, prompts, contrast_prompts, test_prompts, test_contrast_prompts, output_dir)
    result.steps["probe_transfer"] = tr
    test_acc, layer = tr.data["test_accuracy"], tr.data["best_layer"]
    text_best = max(tr.data["embedding_accuracy"], tr.data["text_baseline_accuracy"])
    result.findings.update({
        "layer": layer,
        "test_accuracy": test_acc,
        "test_shuffled_accuracy": tr.data["shuffled_accuracy"],
        "embedding_accuracy": tr.data["embedding_accuracy"],
        "text_baseline_accuracy": tr.data["text_baseline_accuracy"],
    })
    result.evidence.append(
        f"Layer {layer}, chosen on the training texts, scores {test_acc:.0%} on the held-out texts "
        f"(shuffled labels: {tr.data['shuffled_accuracy']:.0%}).")
    result.evidence.append(
        f"Without the model's processing: embeddings alone score {tr.data['embedding_accuracy']:.0%}, "
        f"a bag-of-words classifier {tr.data['text_baseline_accuracy']:.0%}.")

    if test_acc < TRANSFER_MIN:
        result.verdict = "does_not_transfer"
    elif test_acc - text_best < OVER_TEXT_MARGIN:
        result.verdict = "lexical_only"
    else:
        result.verdict = "monitor_works"
    return result
