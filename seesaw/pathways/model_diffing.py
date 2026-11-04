"""Model diffing: what did a fine-tune (or quantisation, or edit) change, and where in the weights?

Inputs are a base model, a changed model with the same tokenizer, and the
prompts to compare them on (both the ones the change was for and ordinary
ones, to see whether it spread).

Measure → describe → locate:

1. Measure — KL divergence of the changed model's next-token distribution
   from the base's on every prompt. Under KL_MIN on average, nothing changed.
2. Describe — the prompts that changed most, what each model predicts there,
   what amplifying the difference predicts, and the tokens the change made
   more likely overall.
3. Locate (same architecture only) — swap one component's weights from the
   changed model into the base and see how much of the KL that alone closes.

Verdicts:
- no_change     — the models behave the same on these prompts.
- localized     — one component's weights account for most of the change.
- distributed   — no single component accounts for most of it.
- changed       — the behaviour changed, but the architectures differ, so
                  it can't be located by swapping weights.
"""

from __future__ import annotations

from pathlib import Path

from transformer_lens import HookedTransformer

from seesaw.interp.model_diffing import diff

from .result import PathwayResult

# Mean KL (nats) below which the models count as behaving the same.
KL_MIN = 0.01
# One component closing this share of the KL alone makes the change localized.
LOCALIZED_MIN = 0.6


def run(
    model: HookedTransformer,
    other_model: HookedTransformer,
    prompts: list[str],
    output_dir: Path,
) -> PathwayResult:
    """Run the model-diffing pathway.

    Args:
        model: The base model.
        other_model: The changed model, with the same tokenizer.
        prompts: Prompts to compare them on.
        output_dir: Where every step saves its plots.

    Returns:
        PathwayResult with per-prompt KL, the most changed prompts, the
        favoured tokens, each component's share of the change, and a verdict.
    """
    result = PathwayResult(pathway="model_diffing")
    d = diff.compute(model, other_model, prompts, output_dir)
    result.steps["model_diff"] = d
    mean_kl = d.data["mean_kl"]
    result.findings.update({"mean_kl": mean_kl, "most_changed": d.data["most_changed"][:5],
                            "favoured_tokens": d.data["favoured_tokens"][:5]})
    if mean_kl < KL_MIN:
        return result.stop("no_change", f"mean KL {mean_kl:.4f} nats is under {KL_MIN}")

    top = d.data["most_changed"][0]
    result.evidence.append(
        f"The changed model's predictions differ from the base's by {mean_kl:.3f} nats on average; "
        f"most on {top['prompt']!r} (KL {top['kl']:.2f}), where the base predicts {top['top_a']!r} "
        f"and the changed model {top['top_b']!r}.")
    result.evidence.append(
        "Tokens the change made more likely: "
        + ", ".join(repr(t["token"]) for t in d.data["favoured_tokens"][:5]) + ".")

    closed = d.data["kl_closed_by_component"]
    if closed is None:
        result.verdict = "changed"
        result.evidence.append("The architectures differ, so the change can't be located by swapping weights.")
        return result
    ranked = sorted(closed.items(), key=lambda kv: -kv[1])
    result.findings["kl_closed_by_component"] = dict(ranked)
    name, share = ranked[0]
    result.evidence.append(
        f"Swapping in the changed model's {name} alone closes {share:.0%} of the KL; next: "
        + ", ".join(f"{n} {s:.0%}" for n, s in ranked[1:4]) + ".")
    result.verdict = "localized" if share >= LOCALIZED_MIN else "distributed"
    return result
