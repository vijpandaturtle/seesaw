"""The Lens tool catalog: every tool's name, what kind of evidence it gives, and one line on what it does.

The single list Scout's plan template, Quill's follow-up prompt, Lens's own
prompts and the eval graders all read, so adding a tool to Lens means adding it
here once rather than in five hand-kept copies. No torch import: Scout, Quill
and the graders run without the experiment stack.

There is one catalog per interp backend, chosen with SEESAW_INTERP_BACKEND:
`seesaw` (default) runs seesaw.interp on TransformerLens text models; `lever`
runs seesaw.vlm on vision-language models through lever. Everything that lists
tools reads the active backend's catalog, so Scout never plans a GPT-2 circuit
experiment Lens can't run on LLaVA, or the reverse.

tests/test_lens_tools.py checks each catalog matches Lens's registry for that backend exactly.

Kinds:
- descriptive — shows what the model represents or attends to; no causal claim.
- estimate    — a gradient-based approximation of a causal effect. Ranks
                candidates cheaply; never report its numbers as effects.
- causal      — an intervention: changes the model and measures what follows.
"""

from __future__ import annotations

import os
from typing import Literal, NamedTuple

Kind = Literal["descriptive", "estimate", "causal"]
Backend = Literal["seesaw", "lever"]
BACKENDS: tuple[Backend, ...] = ("seesaw", "lever")


class LensTool(NamedTuple):
    kind: Kind
    summary: str


LENS_TOOLS: dict[str, LensTool] = {
    # ── descriptive ──────────────────────────────────────────────────────────
    "answer_scores": LensTool(
        "descriptive", "the answer token's probability on each prompt, and its logit diff over negative_tokens: "
                       "does the model do the task at all?"),
    "logit_lens": LensTool(
        "descriptive", "project the residual stream at each layer to vocabulary: where the prediction forms"),
    "patchscope": LensTool(
        "descriptive", "decode each layer's last-token hidden state by patching it into an identity prompt, beside "
                       "the logit lens: where the answer is encoded, read by the model itself"),
    "attention_pattern": LensTool(
        "descriptive", "attention weights per head per layer: what each head looks at"),
    "direct_logit_attribution": LensTool(
        "descriptive", "split the final logit diff into per-head and per-MLP direct contributions"),
    "linear_probe": LensTool(
        "descriptive", "is a concept linearly decodable from the residual stream, and from which layer "
                       "(prompts vs counterfactual_prompts)"),
    "probe_transfer": LensTool(
        "descriptive", "train a probe on prompts vs counterfactual_prompts and score it on held-out test_prompts vs "
                       "test_counterfactual_prompts, against shuffled labels, embeddings alone and bag-of-words"),
    "feature_search": LensTool(
        "descriptive", "which SAE features fire on the prompts, with Neuronpedia's explanations; no gradients"),
    "feature_lookup": LensTool(
        "descriptive", "Neuronpedia's explanation, promoted and suppressed tokens for named SAE `features` at a "
                       "layer"),
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
    "mlp_patching": LensTool(
        "causal", "patch each MLP layer's output from clean into corrupted, one layer at a time"),
    "kv_head_patching": LensTool(
        "causal", "patch each key/value head's keys, values and both from clean into corrupted; in grouped-query "
                  "models (Llama, Qwen, Gemma) a KV head serves a group of query heads"),
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
    "feature_steering": LensTool(
        "causal", "push the residual stream along one SAE `feature`'s decoder direction at its layer and sweep "
                  "the strength, vs random directions, with a loss check"),
    "leace_erasure": LensTool(
        "causal", "erase what separates prompts from counterfactual_prompts (LEACE, every residual site, fitted on "
                  "held-out pairs): does the attribute's pull go, and does the task survive? vs random erasers"),
    "causal_tracing": LensTool(
        "causal", "noise the subject's embeddings (needs `subjects`), restore residual / MLP / attention at each "
                  "layer and position: where a fact is retrieved"),
    "rome_edits": LensTool(
        "causal", "rewrite each fact to target_tokens with ROME at each layer, scored on efficacy, paraphrase and "
                  "specificity: where a fact can be edited"),
    "model_diff": LensTool(
        "causal", "compare the model with other_model (a fine-tune or edit of it): KL per prompt, tokens it "
                  "favours, and which component's weights account for the change"),
}


# Vision-language models through lever (seesaw.vlm). Every tool takes `images`,
# one per prompt; CLIP tools take candidate labels as their prompts.
LEVER_TOOLS: dict[str, LensTool] = {
    # ── descriptive ──────────────────────────────────────────────────────────
    "logit_lens": LensTool(
        "descriptive", "decode the LM residual after each layer: where the answer forms at the answer position, "
                       "and what image tokens decode to"),
    "clip_direct_effects": LensTool(
        "descriptive", "CLIP only: split the image embedding into per-head and per-MLP direct effects on the "
                       "predicted label (prompts are the candidate labels)"),
    # ── causal ───────────────────────────────────────────────────────────────
    "vision_ablation": LensTool(
        "causal", "zero / mean-ablate everything the vision tower hands the LM: how much of the answer needs the "
                  "image, and what the language prior alone says"),
    "image_token_ablation": LensTool(
        "causal", "wipe the image tokens after each LM layer: by which layer has the answer read what it needs "
                  "from the image"),
    "activation_patching": LensTool(
        "causal", "patch clean LM residuals into a corrupted-image run (corrupted_images), per layer, at image / "
                  "text / answer positions"),
    "clip_head_ablation": LensTool(
        "causal", "CLIP only: zero named heads, each alone and all together, and measure the drop in the "
                  "predicted label's probability and the flip rate"),
}

_CATALOGS: dict[str, dict[str, LensTool]] = {"seesaw": LENS_TOOLS, "lever": LEVER_TOOLS}

# The model a job targets when none is named: the one every tool covers,
# feature_explain and circuit_tracing included (Gemma Scope SAEs, Neuronpedia
# graphs). Gated on Hugging Face, so the GPU app needs an HF token.
DEFAULT_MODEL = "google/gemma-2-2b"

# What Scout and Lens's planner are told about models and inputs, per backend.
PLANNING_NOTES: dict[str, str] = {
    "seesaw": (
        "Experiments run with TransformerLens on Modal GPUs. Target google/gemma-2-2b (the default: "
        "every tool works on it), meta-llama/Llama-3.2-1B or -3B, or Qwen/Qwen2.5-0.5B to -7B. The SAE "
        "tools need published SAEs: Gemma-2-2B (Gemma Scope, every layer) and Qwen/Qwen2.5-7B-Instruct "
        "(every 4th layer) only. circuit_tracing reads Neuronpedia graphs for gemma-2-2b and the models "
        "it lists. Llama, Qwen and Gemma use grouped-query attention: heads in a group share keys and "
        "values, so use kv_head_patching for keys and values."
    ),
    "lever": (
        "Experiments run through lever on vision-language models, as Hugging Face checkpoints: "
        "llava-hf/llava-interleave-qwen-0.5b-hf (small, runs anywhere), Qwen/Qwen2.5-VL-3B-Instruct, "
        "or CLIP (openai/clip-vit-base-patch32) for the clip_ tools. Every experiment needs `images`, one per "
        "prompt. An image is a file path, an http(s) URL, `text:<words>` (words on a white canvas), "
        "`solid:<colour>`, or any of those plus `+text:<words>` to stamp words on it (a typographic attack). "
        "Prompts are questions about the image; answer tokens have no leading space (e.g. 'Yes' / 'No'). "
        "For CLIP, prompts are the candidate labels. activation_patching also needs corrupted_images: the same "
        "images with one thing changed so the answer changes."
    ),
}


# Nominate-then-verify chains, per backend: what follows what.
TOOL_CHAINS: dict[str, str] = {
    "seesaw": (
        "attribution_patching or direct_logit_attribution -> head_patching or ablation with `heads` -> "
        "circuit_evaluation; edge_attribution_patching -> path_patching; counterfactual -> linear_probe -> "
        "directional_ablation or steering_vector or leace_erasure; feature_attribution -> feature_clamping, "
        "feature_lookup, feature_steering; answer_scores -> causal_tracing -> rome_edits for facts; linear_probe -> probe_transfer; "
        "attribution_patching -> mlp_patching for MLPs; head_patching -> kv_head_patching on grouped-query models; "
        "logit_lens -> patchscope where the logit lens reads nothing. "
        "circuit_tracing reads attribution graphs from Neuronpedia for gemma-2-2b only."
    ),
    "lever": (
        "vision_ablation first (does the answer need the image at all?) -> image_token_ablation (by which "
        "layer is it read?) -> activation_patching with corrupted_images (which layers and positions carry "
        "it); logit_lens alongside, as description. For CLIP: clip_direct_effects -> clip_head_ablation "
        "with `heads`."
    ),
}


def active_backend() -> Backend:
    """SEESAW_INTERP_BACKEND, defaulting to seesaw. Read per call, so tests can switch it."""
    backend = (os.environ.get("SEESAW_INTERP_BACKEND") or "seesaw").strip().lower()
    if backend not in BACKENDS:
        raise ValueError(f"SEESAW_INTERP_BACKEND={backend!r}; expected one of {BACKENDS}")
    return backend  # type: ignore[return-value]


def catalog(backend: Backend | None = None) -> dict[str, LensTool]:
    """The tool catalog for `backend`, or for the active backend."""
    return _CATALOGS[backend or active_backend()]


def planning_notes(backend: Backend | None = None) -> str:
    return PLANNING_NOTES[backend or active_backend()]


def tool_chains(backend: Backend | None = None) -> str:
    return TOOL_CHAINS[backend or active_backend()]


def names(kind: Kind | None = None, backend: Backend | None = None) -> list[str]:
    """Tool names, optionally of one kind, in catalog order."""
    return [n for n, t in catalog(backend).items() if kind is None or t.kind == kind]


def markdown_table(backend: Backend | None = None) -> str:
    """| Tool | Kind | What it does | — for prompts that list the toolset."""
    rows = ["| Tool | Kind | What it does |", "|---|---|---|"]
    rows += [f"| `{n}` | {t.kind} | {t.summary} |" for n, t in catalog(backend).items()]
    return "\n".join(rows)


def bullet_list(backend: Backend | None = None) -> str:
    """- `name` (kind) — summary"""
    return "\n".join(f"- `{n}` ({t.kind}) — {t.summary}" for n, t in catalog(backend).items())
