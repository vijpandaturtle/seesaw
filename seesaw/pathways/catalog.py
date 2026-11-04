"""What each pathway takes as input, for running one inside a job.

The job store, the worker and the dashboard all need to know a pathway's
inputs without importing torch, so they live here rather than being read off
each module's `run()` signature. `run_from_params` is the one place that does
import the experiment stack, and only when a job actually runs.

Every pathway is given aligned lists: one positive/negative token per prompt,
and one contrast prompt per prompt where it takes them. `validate` checks
that; `check_inputs` also checks every answer token is one token for the
model.

A research job either names a pathway or asks Scout to choose one (AUTO).
A named pathway is binding: Scout only chooses its inputs. `scout_brief`
tells Scout what they are, and `extract_inputs` reads them back out of the
saved plan and checks them. Under AUTO, `choice_brief` lists every pathway
and Scout picks one (or none, if nothing fits) along with its inputs and a
reason; `extract_choice` reads that back. Either way the choice and inputs
stop at the human gate, and a rerun keeps them fixed. A rerun checks the
saved inputs again against its own model before queueing.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple


class Pathway(NamedTuple):
    summary: str
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()
    # Optional inputs that must line up one-to-one with `prompts` when given.
    aligned_optional: tuple[str, ...] = ()
    # Required inputs that are one string rather than a list (a model name).
    text_inputs: tuple[str, ...] = ()
    # Bump when a change to the pathway's steps or thresholds would change its
    # verdict on the same inputs: results are only comparable within a version.
    version: str = "1"


PATHWAYS: dict[str, Pathway] = {
    "circuit_discovery": Pathway(
        "what mechanism does the model use for this task? nominate heads, verify heads and MLPs by "
        "patching, check the circuit is faithful",
        required=("prompts", "positive_tokens", "negative_tokens"),
        optional=("corrupted_prompts", "reference_prompts"),
        aligned_optional=("corrupted_prompts", "reference_prompts"),
        version="2",
    ),
    "concept_direction": Pathway(
        "is a concept a direction the model uses, and can it be steered safely? fit the direction "
        "on half the pairs, remove and steer on the other half, check it can be baked into the weights",
        required=("prompts", "counterfactual_prompts", "positive_tokens", "negative_tokens"),
        optional=("control_prompts", "control_texts"),
    ),
    "attribute_audit": Pathway(
        "the concept_direction pathway under its older name, kept so earlier jobs can be rerun",
        required=("prompts", "counterfactual_prompts", "positive_tokens", "negative_tokens"),
        optional=("control_prompts", "control_texts"),
        version="2",
    ),
    "probe_monitor": Pathway(
        "would a probe on the activations make a good detector for X? train on some texts, test on "
        "held-out ones, compare with text-only baselines",
        required=("prompts", "contrast_prompts", "test_prompts", "test_contrast_prompts"),
    ),
    "spurious_audit": Pathway(
        "does the answer depend on an attribute it shouldn't, and can erasing the attribute (LEACE) "
        "remove that without breaking the task?",
        required=("prompts", "counterfactual_prompts", "positive_tokens", "negative_tokens"),
        optional=("control_texts",),
    ),
    "factual_recall": Pathway(
        "where does the model retrieve these facts? causal tracing over layers and positions",
        required=("prompts", "subjects", "positive_tokens"),
    ),
    "editing_comparison": Pathway(
        "is the layer where a fact is traced the best layer to edit it? ROME at every layer vs tracing",
        required=("prompts", "subjects", "positive_tokens", "target_tokens", "paraphrase_prompts",
                  "neighborhood_prompts", "neighborhood_tokens"),
    ),
    "model_diffing": Pathway(
        "what did a fine-tune (or quantisation, or edit) change, and in which weights? compares the "
        "job's model with other_model",
        required=("prompts",),
        text_inputs=("other_model",),
    ),
    "feature_explain": Pathway(
        "which SAE features drive this output, and what are they? sweeps GPT-2 small SAEs at several "
        "layers (gpt2 only)",
        required=("prompts", "positive_tokens", "negative_tokens"),
        version="2",
    ),
}

# Inputs that have to line up with `prompts`, entry for entry.
_ALIGNED = ("positive_tokens", "negative_tokens", "counterfactual_prompts", "subjects",
            "target_tokens", "paraphrase_prompts", "neighborhood_prompts", "neighborhood_tokens")

# Inputs that are answer tokens, scored as one vocab id each.
_TOKENS = ("positive_tokens", "negative_tokens", "target_tokens", "neighborhood_tokens")

# What each input means, for Scout choosing them. Pathways share names, so one
# table covers all three.
INPUTS: dict[str, str] = {
    "prompts": "prompts the behaviour shows up on; each ends right before the answer token",
    "positive_tokens": "the correct (or attribute-consistent) next token, one per prompt",
    "negative_tokens": "the distractor (or attribute-opposite) next token, one per prompt",
    "corrupted_prompts": "the same prompt with the key information changed (e.g. IOI with the "
                         "repeated name swapped), one per prompt, same token length",
    "reference_prompts": "prompts from a neutral distribution (e.g. IOI's ABC prompts) used as "
                         "ablation means, one per prompt; defaults to the prompts",
    "counterfactual_prompts": "the same prompt with the attribute swapped (e.g. 'nurse' -> "
                              "'engineer'), one per prompt",
    "control_prompts": "attribute-free prompts ('The person said that') showing the model's "
                       "baseline preference; any number",
    "control_texts": "ordinary text for checking an edit doesn't damage the model; any number",
    "contrast_prompts": "training texts without the property the probe should detect; any number",
    "test_prompts": "held-out texts with the property, ideally from a different source or topic "
                    "than `prompts`; any number",
    "test_contrast_prompts": "held-out texts without it, from the same source as `test_prompts`; any number",
    "subjects": "the subject of each factual prompt, written exactly as it appears in the prompt "
                "(e.g. 'The Eiffel Tower'), one per prompt",
    "target_tokens": "the new answer an edit should write in for each fact (e.g. ' Rome'), one per prompt",
    "paraphrase_prompts": "a rewording of each prompt with the same subject and answer, one per prompt",
    "neighborhood_prompts": "for each fact, a prompt about a different subject with the same relation "
                            "(e.g. 'The Colosseum is located in the city of'), one per prompt",
    "neighborhood_tokens": "that neighbourhood prompt's true answer, one per prompt",
    "other_model": "the changed model to compare with the job's model: a TransformerLens model "
                   "name with the same tokenizer (one string, not a list)",
}

# feature_explain's SAEs: GPT-2 small residual SAEs at four layers, swept.
# Layer 8 is the one the known-answer test uses. The feature tools load the
# SAE for each layer (find_sae picks SAE_RELEASE for GPT-2 small).
SAE_RELEASE = "gpt2-small-res-jb"
SAE_LAYERS = (4, 6, 8, 10)
SAE_IDS = tuple(f"blocks.{layer}.hook_resid_pre" for layer in SAE_LAYERS)
SAE_MODELS = ("gpt2", "gpt2-small")


def names() -> list[str]:
    return list(PATHWAYS)


def validate(name: str, params: dict, model: str = "gpt2") -> list[str]:
    """Everything wrong with these inputs, or [] when the job can run."""
    pathway = PATHWAYS.get(name)
    if pathway is None:
        return [f"unknown pathway {name!r}; expected one of {names()}"]
    if not isinstance(params, dict):
        return ["params must be an object"]

    errors: list[str] = []
    allowed = set(pathway.required) | set(pathway.optional) | set(pathway.text_inputs)
    for key in sorted(set(params) - allowed):
        errors.append(f"{name} does not take {key!r}")

    for key in pathway.required:
        value = params.get(key)
        if not _is_str_list(value) or not value:
            errors.append(f"{key} must be a non-empty list of strings")
    for key in pathway.optional:
        if key in params and params[key] is not None and not _is_str_list(params[key]):
            errors.append(f"{key} must be a list of strings")
    for key in pathway.text_inputs:
        if not isinstance(params.get(key), str) or not params[key].strip():
            errors.append(f"{key} must be a non-empty string")

    prompts = params.get("prompts")
    if _is_str_list(prompts) and prompts:
        for key in (*_ALIGNED, *pathway.aligned_optional):
            value = params.get(key)
            if _is_str_list(value) and value and len(value) != len(prompts):
                errors.append(f"{key} has {len(value)} entries but prompts has {len(prompts)}")

    if name == "feature_explain" and model not in SAE_MODELS:
        errors.append(f"feature_explain only runs on gpt2 (its SAE is {SAE_RELEASE}), not {model!r}")
    return errors


def check_inputs(name: str, params: dict, model: str = "gpt2") -> list[str]:
    """Everything stopping these inputs running on this model, tokenizer included."""
    return validate(name, params, model) or token_errors(params, model)


def version(name: str) -> str:
    return PATHWAYS[name].version


# A job's pathway param that asks Scout to choose the pathway.
AUTO = "auto"


def _describe(key: str, kind: str) -> str:
    return f"- `{key}` ({kind}): {INPUTS[key]}."


def _input_rules(model: str) -> list[str]:
    return [
        "Rules for the inputs:",
        "- Every value is a list of strings, except a pathway's single-string inputs (marked",
        "  below). \"One per prompt\" means exactly as many entries as `prompts`, in the same order.",
        "- Answer tokens need a leading space (\" Mary\", not \"Mary\") and must each be a",
        f"  single token in {model}'s tokenizer; prefer short, common words.",
        "- Each prompt should end right before the answer, so the answer is the next token.",
        "- Use at least 8 prompts where the task allows it; vary names and wording.",
    ]


def choice_brief(model: str = "gpt2") -> str:
    """Tell Scout it may choose one of the pathways for this job, or none.

    Scout writes its choice as one fenced json block: {"pathway": name or
    null, "why": ..., "params": {...}}, which `extract_choice` reads back.
    """
    lines = [
        "## Choose a pathway (or none)",
        "",
        "A pathway is a hand-coded investigation with a fixed verdict, checked against known",
        f"results. If one of these answers the question on {model}, choose it: it runs first,",
        "exactly on the inputs you give, and is the job's primary evidence. You choose its inputs",
        "and say why it fits. If none answers the question, choose none and plan the experiments",
        "yourself; don't bend the question to fit a pathway.",
        "",
    ]
    for name, pathway in PATHWAYS.items():
        if name == "attribute_audit":
            continue                  # an older name for concept_direction
        if name == "feature_explain" and model not in SAE_MODELS:
            continue
        needs = ", ".join(f"`{k}`" for k in pathway.required)
        needs += "".join(f", `{k}` (one string)" for k in pathway.text_inputs)
        end = "" if pathway.summary.endswith("?") else "."
        lines.append(f"- `{name}`: {pathway.summary}{end} Needs {needs}"
                     + (f"; optional {', '.join(f'`{k}`' for k in pathway.optional)}" if pathway.optional else "")
                     + ".")
    used = dict.fromkeys(k for pw in PATHWAYS.values()
                         for k in (*pw.required, *pw.text_inputs, *pw.optional))
    lines += ["", "What the inputs mean:", *(_describe(k, "input") for k in used), "",
              *_input_rules(model), "",
              "Put your choice in the plan you save, as exactly one fenced ```json block:",
              "",
              "```json",
              json.dumps({"pathway": "<name>", "why": "<one sentence: why it answers the question>",
                          "params": {"prompts": ["..."], "...": ["..."]}}),
              "```",
              "",
              'or, if no pathway fits: {"pathway": null, "why": "<why none fits>"}.',
              "",
              "The block is checked before the job continues; if it is invalid you will be told why",
              "and asked to save a corrected plan."]
    return "\n".join(lines)


def scout_brief(name: str, model: str = "gpt2") -> str:
    """Tell Scout which pathway will run and what it has to choose.

    The pathway is fixed: Scout's job is only to pick its inputs, written as
    one fenced json block in the plan, which `extract_inputs` reads back.
    """
    pathway = PATHWAYS[name]
    describe = _describe

    lines = [
        f"## Required pathway: `{name}` (version {pathway.version})",
        "",
        f"This job must run the hand-coded `{name}` pathway on {model}: {pathway.summary}.",
        "Lens runs it first, exactly as you specify it, so the result is comparable across",
        "models. You do not design its steps; you choose its inputs. You may still propose",
        "extra experiments in the plan; they will be run after the pathway and labelled extras.",
        "",
        "Inputs:",
        *(describe(k, "required") for k in (*pathway.required, *pathway.text_inputs)),
        *(describe(k, "optional") for k in pathway.optional),
        "",
        *_input_rules(model),
    ]
    if pathway.text_inputs:
        lines.append("- " + ", ".join(f"`{k}`" for k in pathway.text_inputs) + " is a single string, not a list.")
    if name == "feature_explain":
        lines.append(f"- This pathway only runs on gpt2 (its SAE is {SAE_RELEASE}).")
    lines += [
        "",
        "Put the inputs in the plan you save, as exactly one fenced ```json block of the form:",
        "",
        "```json",
        json.dumps({"pathway": name, "params": {**{k: ["..."] for k in pathway.required},
                                                 **{k: "..." for k in pathway.text_inputs}}}),
        "```",
        "",
        "The block is checked before the job continues; if it is invalid you will be told why",
        "and asked to save a corrected plan.",
    ]
    return "\n".join(lines)


_JSON_BLOCK = re.compile(r"```json\s*\n(.*?)\n\s*```", re.DOTALL)


def _pathway_blocks(plan_text: str) -> tuple[list[dict], list[str]]:
    """Every ```json block in the plan that names a pathway (key present, even if null)."""
    blocks = []
    for raw in _JSON_BLOCK.findall(plan_text):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            return [], [f"the ```json block is not valid JSON: {exc}"]
        if isinstance(value, dict) and "pathway" in value:
            blocks.append(value)
    return blocks, []


def extract_choice(plan_text: str, model: str = "gpt2") -> tuple[str | None, dict | None, str, list[str]]:
    """Read the pathway Scout chose (or declined) under AUTO, with its inputs, and check them.

    Returns:
        (name, params, why, []) for a usable choice; (None, None, why, []) when
        Scout chose none; (None, None, "", errors) when the block can't be used.
    """
    blocks, errors = _pathway_blocks(plan_text)
    if errors:
        return None, None, "", errors
    if not blocks:
        return None, None, "", ['no ```json block with {"pathway": ..., "why": ...} found in the plan']
    if len(blocks) > 1:
        return None, None, "", [f"found {len(blocks)} pathway ```json blocks; write exactly one"]
    block = blocks[0]
    name, why = block.get("pathway"), str(block.get("why") or "").strip()
    if not why:
        return None, None, "", ['the block needs a "why": one sentence on why this pathway (or none) fits']
    if name is None:
        return None, None, why, []
    if name not in PATHWAYS or name == "attribute_audit":
        choices = [n for n in names() if n != "attribute_audit"]
        return None, None, "", [f"unknown pathway {name!r}; choose one of {choices}, or null"]
    errors = check_inputs(name, block.get("params") or {}, model)
    return (None, None, "", errors) if errors else (name, block["params"], why, [])


def extract_inputs(name: str, plan_text: str, model: str = "gpt2") -> tuple[dict | None, list[str]]:
    """Read the pathway inputs Scout wrote into its plan, and check them.

    Returns:
        (params, []) when they can run, else (None, everything wrong with them).
    """
    blocks = []
    for raw in _JSON_BLOCK.findall(plan_text):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            return None, [f"the ```json block is not valid JSON: {exc}"]
        if isinstance(value, dict) and "params" in value:
            blocks.append(value)
    if not blocks:
        return None, ['no ```json block with {"pathway": ..., "params": {...}} found in the plan']
    if len(blocks) > 1:
        return None, [f"found {len(blocks)} pathway ```json blocks; write exactly one"]

    block = blocks[0]
    if block.get("pathway") != name:
        return None, [f"the block names pathway {block.get('pathway')!r}, but this job runs {name!r}"]
    params = block["params"]
    errors = check_inputs(name, params, model)
    return (None, errors) if errors else (params, [])


def without_inputs(plan_text: str) -> str:
    """The plan with its pathway ```json block replaced by a one-line note.

    For Lens, once the pathway has run: left in, its prompts read like one
    more experiment to run. Other json blocks are left alone.
    """
    def replace(match: re.Match) -> str:
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError:
            return match.group(0)
        if isinstance(value, dict) and "params" in value:
            return "_(Pathway inputs removed: the pathway has already been run on them.)_"
        if isinstance(value, dict) and "pathway" in value:
            return "_(Pathway choice removed: Scout chose not to run a pathway.)_"
        return match.group(0)

    return _JSON_BLOCK.sub(replace, plan_text)


def token_errors(params: dict, model: str) -> list[str]:
    """Answer tokens that aren't one token for this model's tokenizer.

    Loads only the tokenizer, not the weights. Returns [] (skipping the
    check) when the tokenizer can't be loaded, e.g. offline: the pathway
    itself still refuses multi-token answers when it runs.
    """
    tokenizer = _tokenizer(model)
    if tokenizer is None:
        return []
    errors = []
    for key in _TOKENS:
        for token in params.get(key) or []:
            ids = tokenizer.encode(token, add_special_tokens=False)
            if len(ids) != 1:
                pieces = tokenizer.convert_ids_to_tokens(ids)
                errors.append(f"{key} entry {token!r} is {len(ids)} tokens for {model} ({pieces}); "
                              "pick a single-token word, with a leading space")
    return errors


@lru_cache(maxsize=8)
def _tokenizer(model: str):
    try:
        from transformer_lens.loading_from_pretrained import get_official_model_name
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(get_official_model_name(model))
    except Exception as exc:                         # noqa: BLE001 — the check is best-effort
        print(f"   (skipping single-token check: no tokenizer for {model}: {exc})")
        return None


def run_from_params(name: str, model_name: str, params: dict, output_dir: Path):
    """Run the pathway by model name. Imports torch.

    Every step goes through Lens's tool registry by model name (see
    seesaw.pathways.tools): with SEESAW_LENS_BACKEND=modal each step runs on
    the GPU app, as Lens's experiments do; otherwise in this process on
    Lens's cached copy of the model.
    """
    problems = validate(name, params, model_name)
    if problems:
        raise ValueError("; ".join(problems))

    import importlib

    module = importlib.import_module(f"seesaw.pathways.{name}")
    kwargs = {k: v for k, v in params.items() if v}
    if name == "feature_explain":
        return module.run(model_name, list(SAE_LAYERS), output_dir=output_dir, **kwargs)
    return module.run(model_name, output_dir=output_dir, **kwargs)


def _is_str_list(value) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)
