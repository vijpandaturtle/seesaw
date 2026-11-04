"""How strong each result is, in plain words, and whether its controls ran.

Seesaw is meant to be trusted by people who can't spot an interpretability
illusion themselves: a probe that separates any two prompt sets, a steering
vector that "works" by breaking the model. So every result carries an
assessment that says, without jargon:

- strength — Correlational, Estimate or Causal (from the tool's kind in
  shared/lens_tools.py), with what that kind of result shows and doesn't;
- controls — the baselines the result needs before its number means anything
  (random directions, shuffled labels, a loss check), and whether each ran;
- caveats — the tool's known failure modes, and a warning when there are too
  few prompts to say anything about the behaviour in general.

A result whose required controls didn't run is `complete: False`, and its
label says so. Some tools refuse outright below a minimum prompt count,
where their statistic is meaningless rather than just weak.

No torch import: Lens, Quill, the pathways and the graders all read this.
tests/test_evidence.py checks every catalogued tool has an entry here.
"""

from __future__ import annotations

from typing import NamedTuple

from .lens_tools import Backend, Kind, active_backend, catalog


class Strength(NamedTuple):
    label: str
    shows: str
    does_not_show: str
    # What the interpreting LLM is told about results of this kind.
    how_to_report: str


STRENGTHS: dict[Kind, Strength] = {
    "descriptive": Strength(
        "Correlational",
        "what the model represents or attends to",
        "that the model uses it; that takes an intervention",
        "This tool is descriptive: it shows association, not that anything is necessary. "
        "Do not make causal claims from it.",
    ),
    "estimate": Strength(
        "Estimate",
        "a cheap gradient approximation that ranks candidates",
        "measured effects; test the top candidates with a causal tool",
        "This tool is a gradient-based estimate: its numbers rank candidates but are not "
        "measured effects. Say which candidates deserve a causal test; do not call them causal.",
    ),
    "causal": Strength(
        "Causal",
        "the measured effect of changing the model",
        "that the effect holds beyond these prompts, or that it is the whole mechanism",
        "This tool is a causal intervention: its numbers are measured effects of changing the model.",
    ),
}


class Control(NamedTuple):
    key: str          # the data key the control's result is stored under
    what: str         # what it compares against, in plain words
    required: bool = True


class Trust(NamedTuple):
    controls: tuple[Control, ...] = ()
    caveat: str = ""
    # Below this many prompts the tool refuses: its statistic means nothing.
    min_prompts: int = 1
    # Below this many, the result holds for these prompts only.
    thin_below: int = 8


_PATCHING = ("Effects are fractions of the clean-minus-corrupted gap, so they are only as "
             "meaningful as the corrupted prompts.")
_ALONE = ("A component that only matters alongside others (a backup) looks small when "
          "tested alone; check the joint result.")

TRUST: dict[Backend, dict[str, Trust]] = {
    "seesaw": {
        "logit_lens": Trust(caveat=(
            "Early layers are decoded with the final layer's unembedding, so they can look "
            "empty even when the information is there.")),
        "attention_pattern": Trust(thin_below=1, caveat=(
            "Attention weight is not importance: a head can attend strongly and contribute nothing.")),
        "direct_logit_attribution": Trust(caveat=(
            "Counts only what each component writes straight to the output; it misses components "
            "that act through later ones.")),
        "linear_probe": Trust(
            controls=(Control("shuffled_accuracy", "the same probe trained on shuffled labels"),),
            min_prompts=4,
            caveat="Decodable is not the same as used: follow with directional_ablation."),
        "feature_search": Trust(caveat=(
            "Feature labels are written automatically by Neuronpedia and can be wrong.")),
        "counterfactual": Trust(
            controls=(Control("control_mean_ld", "attribute-free control prompts", required=False),),
            caveat="Shows the output depends on the swap, not which part of the model does it."),
        "attribution_patching": Trust(caveat=_PATCHING),
        "feature_attribution": Trust(caveat=(
            "Feature labels are written automatically by Neuronpedia and can be wrong.")),
        "edge_attribution_patching": Trust(caveat=_PATCHING),
        "circuit_tracing": Trust(thin_below=1, caveat=(
            "Explains one prompt through a replacement model; check the share of influence "
            "through error nodes before trusting it.")),
        "ablation": Trust(caveat=(
            "Zero and mean ablation push activations off-distribution and miss backup heads; "
            "patching is the cleaner test.")),
        "activation_patching": Trust(caveat=_PATCHING),
        "head_patching": Trust(caveat=f"{_PATCHING} {_ALONE}"),
        "circuit_evaluation": Trust(controls=(
            Control("random_faithfulness", "random head sets of the same size, kept"),
            Control("random_necessity", "random head sets of the same size, removed"),
        )),
        "path_patching": Trust(caveat=f"{_PATCHING} {_ALONE}"),
        "steering_vector": Trust(controls=(
            Control("random_abs_slope", "random directions of the same size"),
            Control("control_loss_after", "language-modelling loss with the strongest push"),
        ), caveat="A large push can move the output by damaging the model; compare the loss."),
        "directional_ablation": Trust(controls=(
            Control("random_fraction_removed", "random directions removed the same way"),
            Control("control_loss_after", "language-modelling loss with the direction removed"),
        )),
        "feature_clamping": Trust(controls=(
            Control("random_joint_change", "the same number of random active features"),
        )),
        "answer_scores": Trust(caveat=(
            "Shows how strongly the model gives the answer, not why.")),
        "probe_transfer": Trust(controls=(
            Control("shuffled_accuracy", "shuffled labels"),
            Control("embedding_accuracy", "a probe on the embeddings alone"),
            Control("text_baseline_accuracy", "a bag-of-words classifier with no model"),
        ), min_prompts=4,
            caveat="The held-out texts test transfer only as far as they differ from the training texts."),
        "feature_lookup": Trust(thin_below=1, caveat=(
            "Feature labels are written automatically by Neuronpedia and can be wrong.")),
        "mlp_patching": Trust(caveat=f"{_PATCHING} {_ALONE}"),
        "leace_erasure": Trust(controls=(
            Control("random_fraction_removed", "random erasers of the same rank"),
            Control("control_loss_after", "language-modelling loss with the erasers applied"),
        ), caveat="LEACE removes what a linear readout could see; a nonlinear one might still see it."),
        "causal_tracing": Trust(caveat=(
            "Noised embeddings are unnatural input: recovery shows where clean information helps, "
            "not the whole mechanism.")),
        "rome_edits": Trust(caveat=(
            "The key covariance comes from a short built-in corpus, so specificity is rougher than "
            "published ROME numbers.")),
        "model_diff": Trust(caveat=(
            "Covers only these prompts: a change that shows up on other inputs won't appear here.")),
    },
    "lever": {
        "logit_lens": Trust(caveat=(
            "Early layers are decoded with the final layer's unembedding, so they can look "
            "empty even when the information is there.")),
        "clip_direct_effects": Trust(caveat=(
            "Counts only direct effects on the embedding; it misses components that act "
            "through later ones.")),
        "vision_ablation": Trust(caveat=(
            "Mean and zero ablation push the LM off-distribution; a drop can be damage, not reliance.")),
        "image_token_ablation": Trust(caveat=(
            "Wiping tokens is a blunt edit; a late layer can look necessary because the model breaks.")),
        "activation_patching": Trust(caveat=(
            "Effects are fractions of the clean-minus-corrupted gap, so they are only as "
            "meaningful as the corrupted images.")),
        "clip_head_ablation": Trust(caveat=_ALONE),
    },
}


# Pathway steps that aren't Lens tools: their kind, and what they need.
PATHWAY_STEPS: dict[str, tuple[Kind, Trust]] = {
    "bake_check": ("causal", Trust(caveat=(
        "Checks the edit folded into the weights matches the hook version; it doesn't test the "
        "direction itself."))),
}


def _lookup(tool: str, backend: Backend | None) -> tuple[Kind, Trust]:
    """The kind and trust entry for a Lens tool or pathway step (`name@layer` reads as `name`)."""
    tool = tool.split("@", 1)[0]
    entry = catalog(backend).get(tool)
    if entry is not None:
        return entry.kind, TRUST[backend or active_backend()].get(tool, Trust())
    return PATHWAY_STEPS.get(tool, ("descriptive", Trust()))


def trust(tool: str, backend: Backend | None = None) -> Trust:
    return _lookup(tool, backend)[1]


def refusal(tool: str, n_prompts: int, backend: Backend | None = None) -> str | None:
    """Why this tool won't run on this few prompts, or None when it will."""
    need = trust(tool, backend).min_prompts
    if n_prompts < need:
        return (f"{tool} needs at least {need} prompts to mean anything; got {n_prompts}. "
                "Add more prompts rather than reading a number off too few.")
    return None


def assess(tool: str, data: dict, n_prompts: int | None = None,
           backend: Backend | None = None) -> dict:
    """The plain-language assessment of one successful result.

    Args:
        tool: Catalog name of the tool that produced `data`, or a pathway step
            name from PATHWAY_STEPS.
        data: The result's data, where controls store their numbers.
        n_prompts: How many prompts it ran on; None skips the thin-data check.
        backend: Defaults to the active one.

    Returns:
        {strength, kind, shows, does_not_show, controls, complete, caveats, label}.
        `complete` is False when a required control didn't run.
    """
    kind, spec = _lookup(tool, backend)
    strength = STRENGTHS[kind]

    controls = [
        {"name": c.key, "what": c.what, "required": c.required,
         "present": data.get(c.key) is not None}
        for c in spec.controls
    ]
    missing = [c["what"] for c in controls if c["required"] and not c["present"]]
    caveats = [spec.caveat] if spec.caveat else []
    thin = n_prompts is not None and n_prompts < spec.thin_below
    if thin:
        caveats.append(f"Only {n_prompts} prompt{'s' if n_prompts != 1 else ''}: this holds for "
                       "these prompts, not for the behaviour in general.")

    label = strength.label
    if missing:
        label += f", incomplete (no comparison against {'; '.join(missing)})"
    elif any(c["required"] for c in controls):
        label += ", controlled"
    if thin:
        label += f", {n_prompts} prompt{'s' if n_prompts != 1 else ''}"

    return {
        "strength": strength.label,
        "kind": kind,
        "shows": strength.shows,
        "does_not_show": strength.does_not_show,
        "controls": controls,
        "complete": not missing,
        "caveats": caveats,
        "label": label,
    }


def as_text(assessment: dict) -> str:
    """The assessment as a few plain lines, for prompts and reports."""
    lines = [f"Evidence: {assessment['label']}. Shows {assessment['shows']}; "
             f"does not show {assessment['does_not_show']}."]
    for c in assessment["controls"]:
        state = "ran" if c["present"] else ("MISSING" if c["required"] else "not run (optional)")
        lines.append(f"- Control, {c['what']}: {state}")
    lines += [f"- Caveat: {c}" for c in assessment["caveats"]]
    return "\n".join(lines)


def how_to_report(tool: str, backend: Backend | None = None) -> str:
    entry = catalog(backend).get(tool)
    return STRENGTHS[entry.kind].how_to_report if entry else ""
