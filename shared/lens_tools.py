"""The Lens tool catalog: every tool's name, what kind of evidence it gives, and one line on what it does.

The single list Scout's plan template, Quill's follow-up prompt, Lens's own
prompts and the eval graders all read, so adding a tool to Lens means adding it
here once rather than in five hand-kept copies. No torch import: Scout, Quill
and the graders run without the experiment stack.

tests/test_lens_tools.py checks this matches Lens's TOOL_REGISTRY exactly.

Kinds:
- descriptive — shows what the model represents or attends to; no causal claim.
- estimate    — a gradient-based approximation of a causal effect. Ranks
                candidates cheaply; never report its numbers as effects.
- causal      — an intervention: changes the model and measures what follows.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

Kind = Literal["descriptive", "estimate", "causal"]


class LensTool(NamedTuple):
    kind: Kind
    summary: str


LENS_TOOLS: dict[str, LensTool] = {
    # ── descriptive ──────────────────────────────────────────────────────────
    "logit_lens": LensTool(
        "descriptive", "project the residual stream at each layer to vocabulary: where the prediction forms"),
    "attention_pattern": LensTool(
        "descriptive", "attention weights per head per layer: what each head looks at"),
    "direct_logit_attribution": LensTool(
        "descriptive", "split the final logit diff into per-head and per-MLP direct contributions"),
    "linear_probe": LensTool(
        "descriptive", "is a concept linearly decodable from the residual stream, and from which layer "
                       "(prompts vs counterfactual_prompts)"),
    "feature_search": LensTool(
        "descriptive", "which SAE features fire on the prompts, with Neuronpedia's explanations; no gradients"),
    "counterfactual": LensTool(
        "descriptive", "swap one attribute in the input (counterfactual_prompts) and measure how far the "
                       "output moves; inputs and outputs only"),
    # ── estimate ─────────────────────────────────────────────────────────────
    "attribution_patching": LensTool(
        "estimate", "every head's and MLP's patching effect estimated from one backward pass; nominates heads "
                    "for head_patching"),
    "feature_attribution": LensTool(
        "estimate", "every SAE feature's contribution to the logit diff (activation x gradient), labelled from "
                    "Neuronpedia; nominates features for feature_clamping"),
    "edge_attribution_patching": LensTool(
        "estimate", "every sender->receiver edge's patching effect (EAP-IG): which connections between heads "
                    "carry the behaviour; nominates edges for path_patching"),
    "circuit_tracing": LensTool(
        "estimate", "the attribution graph for one prompt from Neuronpedia (gemma-2-2b only): top transcoder "
                    "features by influence on the predicted token, and the share through error nodes"),
    # ── causal ───────────────────────────────────────────────────────────────
    "ablation": LensTool(
        "causal", "zero / mean / resample-ablate heads (all, or a `heads` list) and measure the drop in logit diff"),
    "activation_patching": LensTool(
        "causal", "patch the clean residual stream into the corrupted run by layer and position"),
    "head_patching": LensTool(
        "causal", "patch named heads from clean into corrupted, each alone and all together"),
    "circuit_evaluation": LensTool(
        "causal", "how much behaviour a set of heads keeps (faithfulness) and removes (necessity), vs random "
                  "head sets"),
    "path_patching": LensTool(
        "causal", "patch named edges ('L8H6->L9H9.q') from clean into corrupted, each alone and all together"),
    "steering_vector": LensTool(
        "causal", "add the prompts-minus-counterfactuals direction to the counterfactual prompts at a layer and "
                  "sweep its strength"),
    "directional_ablation": LensTool(
        "causal", "remove that direction from the residual stream everywhere: does the gap between prompts and "
                  "counterfactuals close, vs random directions?"),
    "feature_clamping": LensTool(
        "causal", "switch named SAE features off (or set them to a value) and measure the logit diff, vs random "
                  "features"),
}


def names(kind: Kind | None = None) -> list[str]:
    """Tool names, optionally of one kind, in catalog order."""
    return [n for n, t in LENS_TOOLS.items() if kind is None or t.kind == kind]


def markdown_table() -> str:
    """| Tool | Kind | What it does | — for prompts that list the toolset."""
    rows = ["| Tool | Kind | What it does |", "|---|---|---|"]
    rows += [f"| `{n}` | {t.kind} | {t.summary} |" for n, t in LENS_TOOLS.items()]
    return "\n".join(rows)


def bullet_list() -> str:
    """- `name` (kind) — summary"""
    return "\n".join(f"- `{n}` ({t.kind}) — {t.summary}" for n, t in LENS_TOOLS.items())
