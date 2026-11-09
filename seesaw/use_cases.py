"""What Seesaw is for, as AI safety use cases: a job to do, how to score it, and what it has to beat.

A pathway is a method (fit a direction, trace a fact). A use case is a safety job
someone wants done with a model (monitor for deception, find a backdoor, unlearn
hazardous knowledge), scored by an outcome metric and compared with the best way
to do it without interpretability. Each sits in an area: monitor (catch unsafe
behavior at runtime), audit (find a hidden problem before deployment), fix
(remove or steer it), or understand (explain the model, with no outcome to beat).
The pathways are how a use case gets done; one use case can use several.

Each use case says honestly how far Seesaw gets with it:

- `runnable`: a pathway does the job and measures the outcome against its baseline.
- `partial`: a pathway does the job, but the outcome metric or the baseline arm
  isn't there yet. `gap` says what's missing.
- `planned`: no pathway yet. `gap` says what it needs.

The job store, the dashboard and Scout read this without importing torch.
`seesaw use-cases --json PATH` writes it for the dashboard, which keeps a copy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, NamedTuple

from seesaw.pathways.catalog import PATHWAYS

Status = Literal["runnable", "partial", "planned"]
Modality = Literal["text", "vision"]
Area = Literal["monitor", "audit", "fix", "understand"]

STATUSES: tuple[Status, ...] = ("runnable", "partial", "planned")
# What a use case does for safety, in the order the dashboard shows them.
AREAS: dict[Area, str] = {
    "monitor": "Monitor: catch unsafe behavior at runtime",
    "audit": "Audit: find a hidden problem before deployment",
    "fix": "Fix: remove or steer an unsafe behavior",
    "understand": "Understand: explain how the model works, with no outcome to beat",
}


class UseCase(NamedTuple):
    title: str
    # The question someone brings, in their words.
    question: str
    # How a result is scored. For the understand area, what the pathway reports.
    metric: str
    # The best way to do the job without interpretability, which the metric is compared with.
    baseline: str
    status: Status
    # What it does for safety: monitor, audit, fix, or understand (no outcome to beat).
    area: Area
    modality: Modality = "text"
    # The pathways that do the job, the main one first. Empty when planned.
    pathways: tuple[str, ...] = ()
    # lever tools a vision use case would run (shared/lens_tools.py LEVER_TOOLS).
    tools: tuple[str, ...] = ()
    # What stands between the use case and `runnable`. Empty when runnable.
    gap: str = ""
    # A question to start from, ideally one whose answer is already known.
    example: str = ""


USE_CASES: dict[str, UseCase] = {
    # ── Monitor: catch unsafe behavior at runtime ────────────────────────────
    "build_a_safety_monitor": UseCase(
        "Build a safety monitor",
        "Would a monitor on the model's activations flag deception, prompt injection or harmful "
        "intent on new texts, better than the words alone?",
        metric="held-out accuracy of the activation probe, on texts from a different source",
        baseline="bag-of-words and embeddings-only probes on the same texts, and shuffled labels",
        status="runnable",
        area="monitor",
        pathways=("probe_monitor",),
        example="Would a linear probe on the residual stream flag prompt-injection attempts better "
                "than a bag-of-words baseline?",
    ),
    "check_reasoning_faithfulness": UseCase(
        "Check reasoning faithfulness",
        "Does the model's stated reasoning reflect what actually drives its answer?",
        metric="how often the stated reason matches the input the answer causally depends on",
        baseline="counterfactual edits to the prompt, read off the output alone",
        status="planned",
        area="monitor",
        gap="needs a pathway that plants a hint the answer depends on and checks whether the "
            "reasoning mentions it, with attribution to tell what the answer used",
    ),
    # ── Audit: find a hidden problem before deployment ───────────────────────
    "find_failure_modes": UseCase(
        "Find failure modes",
        "Where does my model break or behave unsafely, and can its internals point me there faster?",
        metric="distinct failures found under a fixed query budget, and how severe they are",
        baseline="black-box red-teaming and fuzzing under the same budget",
        status="planned",
        area="audit",
        gap="needs a pathway that searches from the model's internals (push along a direction or "
            "feature, or attack the activations, then find real inputs that do the same) and a "
            "scorer for distinct failures; find_a_backdoor is its version with a planted answer",
    ),
    "find_a_backdoor": UseCase(
        "Find a backdoor",
        "Does this model have a hidden trigger that switches on bad behavior, and what is it?",
        metric="whether the trigger was identified, and in how many queries",
        baseline="black-box red-teaming under the same query budget",
        status="planned",
        area="audit",
        gap="needs model organisms with a planted trigger (a LoRA on a small model) and a pathway "
            "that searches under a fixed query budget",
    ),
    "audit_hidden_objective": UseCase(
        "Audit a hidden objective",
        "Is this model pursuing a goal it wasn't meant to have?",
        metric="whether the audit names the planted goal or behavior",
        baseline="behavioral testing and prompting",
        status="planned",
        area="audit",
        gap="needs model organisms with a planted objective; Scout and Lens would plan the audit "
            "freely, scored against the plant",
    ),
    "detect_sandbagging": UseCase(
        "Detect sandbagging",
        "Is the model deliberately underperforming on a capability evaluation?",
        metric="accuracy at telling sandbagging model organisms from honest ones, and false alarms",
        baseline="behavioral evals: elicitation with fine-tuning or few-shot prompting",
        status="planned",
        area="audit",
        gap="needs model organisms trained to hide a capability, and a pathway that compares what "
            "the activations know with what the output says",
    ),
    "audit_a_finetune": UseCase(
        "Audit a fine-tune",
        "Did a fine-tune change more than it was meant to, such as making the model broadly "
        "misaligned?",
        metric="recall of a planted change, and false alarms",
        baseline="comparing the two models' outputs on eval sets",
        status="partial",
        area="audit",
        pathways=("model_diffing",),
        gap="model_diffing reports where the models differ; it has no fine-tunes with a planted "
            "change to score recall and false alarms against, and no output-diff baseline",
        example="What did instruction tuning change between Qwen2.5-1.5B and "
                "Qwen2.5-1.5B-Instruct, and in which layers?",
    ),
    # ── Fix: remove or steer an unsafe behavior ──────────────────────────────
    "steer_unsafe_behavior": UseCase(
        "Steer away from unsafe behavior",
        "Can I turn down a behavior like sycophancy or bias without damaging everything else?",
        metric="size of the targeted change, against collateral damage on unrelated text",
        baseline="prompting, and safety fine-tuning",
        status="partial",
        area="fix",
        pathways=("concept_direction",),
        gap="concept_direction measures the change against random directions and checks control "
            "texts; it doesn't yet run a prompting baseline",
        example="Is grammatical gender in occupation words a direction the model uses, and can "
                "removing it stop gendered pronoun predictions?",
    ),
    "remove_spurious_correlation": UseCase(
        "Remove a spurious correlation",
        "Does the answer depend on something it shouldn't, like a name or a dialect, and can I "
        "remove that without breaking the task?",
        metric="accuracy on a balanced or shifted test set after the fix",
        baseline="fine-tuning or reweighting on rebalanced data",
        status="partial",
        area="fix",
        pathways=("spurious_audit",),
        gap="spurious_audit measures the dependence and checks the task survives erasure; it "
            "doesn't yet score a shifted test set or run a rebalanced baseline",
        example="Does sentiment classification depend on the reviewer's name, and can erasing it "
                "keep the task working?",
    ),
    "unlearn_knowledge": UseCase(
        "Edit or unlearn knowledge",
        "Can I remove hazardous knowledge, or correct a fact, and keep the rest of what the "
        "model knows?",
        metric="forgetting on the target (a WMDP-style set), against ability kept elsewhere; for "
               "edits, efficacy and paraphrases against damage to neighboring facts",
        baseline="gradient-ascent unlearning; fine-tuning on the corrected fact",
        status="partial",
        area="fix",
        pathways=("editing_comparison", "factual_recall"),
        gap="editing_comparison scores ROME edits on single facts; it doesn't yet unlearn a body "
            "of knowledge or run a gradient-based baseline",
        example="Is the layer where causal tracing locates a landmark's city the best layer to "
                "edit it with ROME?",
    ),
    # ── Vision models, through lever ─────────────────────────────────────────
    "resist_image_injection": UseCase(
        "Resist prompt injection in images",
        "Does text written in an image override what the image shows, or what the user asked?",
        metric="accuracy on attacked images, against accuracy on clean ones",
        baseline="prompting, and a light fine-tune on attacked images",
        status="planned",
        area="fix",
        modality="vision",
        tools=("clip_direct_effects", "clip_head_ablation", "activation_patching"),
        gap="the lever tools can find and ablate the heads that read the text; there's no "
            "pathway or attacked test set yet",
        example="Which CLIP heads read text stamped on an image, and does ablating them stop "
                "the label flipping?",
    ),
    "reduce_hallucination": UseCase(
        "Reduce object hallucination",
        "Is my VLM answering from the image, or making things up from what the language model "
        "expects?",
        metric="accuracy on yes/no object questions (POPE-style)",
        baseline="prompting, and contrastive decoding",
        status="planned",
        area="fix",
        modality="vision",
        tools=("vision_ablation", "image_token_ablation", "logit_lens"),
        gap="the lever tools measure how much an answer needs the image; there's no pathway yet "
            "that intervenes and scores hallucination",
        example="How much of LLaVA's answer to 'Is there a dog in the image?' comes from the "
                "image rather than the language prior?",
    ),
    "remove_spurious_background": UseCase(
        "Remove a spurious background",
        "Is my image classifier using the background instead of the object?",
        metric="worst-group accuracy on a balanced test set (Waterbirds)",
        baseline="group reweighting, and retraining the linear head",
        status="planned",
        area="fix",
        modality="vision",
        tools=("clip_direct_effects", "clip_head_ablation"),
        gap="the lever tools nominate and ablate CLIP heads; there's no pathway yet that picks "
            "heads and scores worst-group accuracy",
        example="Which CLIP heads carry the background in Waterbirds, and does ablating them "
                "raise worst-group accuracy?",
    ),
    # ── Understand: explain how the model works (no outcome to beat) ─────────
    "understand_a_capability": UseCase(
        "Understand a capability",
        "How does the model do X?",
        metric="circuit faithfulness: how much of the behavior the circuit recovers",
        baseline="random circuits of the same size",
        status="runnable",
        area="understand",
        pathways=("circuit_discovery",),
        example="Which attention heads and MLPs implement indirect object identification?",
    ),
    "locate_a_fact": UseCase(
        "Locate a fact",
        "Where does the model retrieve a fact?",
        metric="indirect effect of restoring each layer and position",
        baseline="corrupted-run effect",
        status="runnable",
        area="understand",
        pathways=("factual_recall",),
        example="At which layers and token positions does the model recall the city a landmark is in?",
    ),
    "explain_an_output": UseCase(
        "Explain an output",
        "Why did the model say X here? Which features drive it?",
        metric="how much clamping the top SAE features moves the answer",
        baseline="random features at the same layer",
        status="runnable",
        area="understand",
        pathways=("feature_explain",),
        example="Which SAE features drive the model to complete a famous quote?",
    ),
}


def names() -> list[str]:
    return list(USE_CASES)


def for_pathway(pathway: str | None) -> str | None:
    """The use case a pathway serves: the one that lists it first, else any that lists it."""
    if not pathway:
        return None
    for name, case in USE_CASES.items():
        if case.pathways[:1] == (pathway,):
            return name
    if pathway == "attribute_audit":  # concept_direction's older name
        return for_pathway("concept_direction")
    return next((n for n, c in USE_CASES.items() if pathway in c.pathways), None)


def main_pathway(name: str) -> str | None:
    """The pathway a job for this use case runs, or None when the use case is planned."""
    case = USE_CASES[name]
    return case.pathways[0] if case.pathways else None


def problems() -> list[str]:
    """What's inconsistent in the catalog; empty when it's sound."""
    from shared.lens_tools import LEVER_TOOLS

    found = []
    for name, case in USE_CASES.items():
        found += [f"{name}: unknown pathway {p}" for p in case.pathways if p not in PATHWAYS]
        found += [f"{name}: unknown lever tool {t}" for t in case.tools if t not in LEVER_TOOLS]
        if case.status == "planned" and case.pathways:
            found.append(f"{name}: planned but lists pathways")
        if case.status != "planned" and not case.pathways:
            found.append(f"{name}: {case.status} but lists no pathway")
        if (case.status == "runnable") == bool(case.gap):
            found.append(f"{name}: a gap goes with partial or planned, and only with those")
        if case.area not in AREAS:
            found.append(f"{name}: unknown area {case.area}")
        if case.modality == "vision" and case.status != "planned" and not case.tools:
            found.append(f"{name}: a vision use case names its lever tools")
    return found


def as_json() -> dict:
    """The catalog as the dashboard reads it."""
    return {
        "statuses": list(STATUSES),
        "areas": [{"name": a, "label": label} for a, label in AREAS.items()],
        "use_cases": [
            {"name": name, **{k: list(v) if isinstance(v, tuple) else v
                              for k, v in case._asdict().items()}}
            for name, case in USE_CASES.items()
        ],
    }


def write_json(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(as_json(), indent=2) + "\n")
    return out
