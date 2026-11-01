import inspect

from .logit_lens import run_logit_lens
from .attention_pattern import run_attention_pattern
from .direct_logit_attribution import run_direct_logit_attribution
from .ablation import run_ablation
from .activation_patching import run_activation_patching
from .attribution_patching import run_attribution_patching
from .head_patching import run_head_patching
from .circuit_evaluation import run_circuit_evaluation
from .counterfactual import run_counterfactual
from .linear_probe import run_linear_probe
from .steering_vector import run_steering_vector
from .directional_ablation import run_directional_ablation
from .features import run_feature_attribution, run_feature_clamping, run_feature_search
from .edges import run_edge_attribution_patching, run_path_patching
from .circuit_tracing import run_circuit_tracing
from .lever_tools import LEVER_REGISTRY

from shared.lens_tools import Backend, active_backend

# Keys must match shared/lens_tools.py, the catalog Scout, Quill and the
# graders read (tests/test_lens_tools.py checks they agree).
SEESAW_REGISTRY: dict = {
    "logit_lens":               run_logit_lens,
    "attention_pattern":        run_attention_pattern,
    "direct_logit_attribution": run_direct_logit_attribution,
    "linear_probe":             run_linear_probe,
    "feature_search":           run_feature_search,
    "counterfactual":           run_counterfactual,
    "attribution_patching":     run_attribution_patching,
    "feature_attribution":      run_feature_attribution,
    "edge_attribution_patching": run_edge_attribution_patching,
    "circuit_tracing":          run_circuit_tracing,
    "ablation":                 run_ablation,
    "activation_patching":      run_activation_patching,
    "head_patching":            run_head_patching,
    "circuit_evaluation":       run_circuit_evaluation,
    "path_patching":            run_path_patching,
    "steering_vector":          run_steering_vector,
    "directional_ablation":     run_directional_ablation,
    "feature_clamping":         run_feature_clamping,
}

_REGISTRIES = {"seesaw": SEESAW_REGISTRY, "lever": LEVER_REGISTRY}


def registry(backend: Backend | None = None) -> dict:
    """The tools Lens runs for `backend`, or for SEESAW_INTERP_BACKEND."""
    return _REGISTRIES[backend or active_backend()]


# The TransformerLens tools. Modal's runner uses this directly: it runs the
# seesaw backend only. Everything else goes through registry().
TOOL_REGISTRY = SEESAW_REGISTRY

# run_experiment supplies these from the spec; the LLM never provides them
# as tool_kwargs.
_NODE_SUPPLIED = ("model", "prompts")

# What each extra argument means, for the parse_plan prompt. Keys must match
# real parameter names — required_tool_kwargs() is derived from the actual
# signatures, so a rename here can't silently drift from the tools.
ARG_HELP = {
    "positive_tokens":   "tokens to boost, e.g. [' Mary'] for IOI or [' she'] for gender bias",
    "negative_tokens":   "tokens to suppress, e.g. [' John'] for IOI or [' he'] for gender bias",
    "corrupted_prompts": "each clean prompt with the key entity swapped so the model gets it "
                         "wrong, e.g. 'When John and Mary...' -> 'When John and John...'",
    "counterfactual_prompts": "each prompt with only the attribute under study swapped, same "
                              "length, e.g. 'The nurse said that' -> 'The engineer said that'",
    "heads":             "heads to test, named 'L<layer>H<head>', e.g. ['L9H9', 'L10H7'] — take "
                         "them from an earlier sweep's top_heads",
    "edges":             "edges to test, written 'sender->receiver', e.g. ['L8H6->L9H9.q', "
                         "'L9H9->logits'] — take them from edge_attribution_patching's top_edges",
    "features":          "SAE feature indices to switch off, e.g. [658, 16308] — take them from "
                         "feature_attribution's top_features, and use its layer",
    "images":            "one image per prompt (or one for all): a file path, an http(s) URL, "
                         "'text:<words>', 'solid:<colour>', or any of those + '+text:<words>' to stamp "
                         "words on it, e.g. ['photo.jpg', 'text:STOP', 'dog.jpg+text:cat']",
    "corrupted_images":  "each image with the one thing under study changed so the answer changes, "
                         "same form as images, e.g. 'dog.jpg' -> 'dog.jpg+text:cat'",
}

# Plausible names an LLM reaches for, mapped to the real parameter. The old
# parse_plan prompt asked for io_tokens/subject_tokens, which no tool accepts.
KWARG_ALIASES = {
    "io_tokens":         "positive_tokens",
    "subject_tokens":    "negative_tokens",
    "correct_tokens":    "positive_tokens",
    "incorrect_tokens":  "negative_tokens",
    "answer_tokens":     "positive_tokens",
    "wrong_tokens":      "negative_tokens",
    "corrupt_prompts":   "corrupted_prompts",
    "corrupted":         "corrupted_prompts",
    "contrast_prompts":  "counterfactual_prompts",
    "swapped_prompts":   "counterfactual_prompts",
    "counterfactuals":   "counterfactual_prompts",
    "heads_to_test":     "heads",
    "circuit":           "heads",
    "feature_indices":   "features",
    "connections":       "edges",
    "edges_to_test":     "edges",
    "image":             "images",
    "image_paths":       "images",
    "counterfactual_images": "corrupted_images",
    "corrupt_images":    "corrupted_images",
}

# The second prompt set goes by two names: patching tools say corrupted_prompts,
# audit tools say counterfactual_prompts. Same idea (the prompts with one thing
# swapped), so whichever the plan used is routed to whichever the tool takes.
_PAIR_ARGS = ("corrupted_prompts", "counterfactual_prompts")

_TOKEN_ARGS = ("positive_tokens", "negative_tokens")
# Per-prompt arguments a single value broadcasts across.
_BROADCAST_ARGS = _TOKEN_ARGS + ("images", "corrupted_images")


def expand_pairings(prompts: list[str], kwargs: dict) -> tuple[list[str], dict]:
    """Turn a plan's separate prompt and image lists into one image per prompt.

    Plans list prompts and images side by side ("these 3 questions, these 8
    images") and mean every pairing, while the lever tools take one image per
    prompt. When the counts differ and there's more than one image, every
    prompt is run on every image, prompt-major: (p1, i1), (p1, i2), … (p3, i8).

    Lists that follow the images (corrupted_images, or tokens with one entry
    per image) are tiled the same way; tokens with one entry per prompt are
    repeated per image. Equal counts are left paired, as before, and a single
    image still broadcasts in normalize_tool_kwargs.

    Returns:
        (prompts, kwargs with aliases resolved). Unchanged when there's
        nothing to expand.
    """
    kwargs = {KWARG_ALIASES.get(k, k): v for k, v in (kwargs or {}).items()}
    images = kwargs.get("images")
    images = [images] if isinstance(images, str) else images
    n, k = len(prompts), len(images or [])
    if k <= 1 or k == n:
        return prompts, kwargs

    out = dict(kwargs, images=list(images) * n)
    for arg in ("corrupted_images",) + _TOKEN_ARGS:
        value = kwargs.get(arg)
        if not isinstance(value, list) or len(value) == 1:
            continue                       # absent, or one value for all
        if len(value) == k:
            out[arg] = value * n           # one per image
        elif len(value) == n:
            out[arg] = [v for v in value for _ in range(k)]   # one per prompt
    return [p for p in prompts for _ in range(k)], out


def required_tool_kwargs(tool_name: str) -> list[str]:
    """Extra arguments a tool needs beyond model and prompts.

    Read off the live signature, so adding a tool or renaming a parameter
    updates both the parse_plan prompt and dispatch validation.
    """
    fn = registry()[tool_name]
    return [
        p.name
        for p in inspect.signature(fn).parameters.values()
        if p.default is inspect.Parameter.empty and p.name not in _NODE_SUPPLIED
    ]


def tool_kwargs_guide() -> str:
    """Per-tool tool_kwargs contract, formatted for the parse_plan prompt.

    Tools first, each with the arguments it requires; then each argument's
    meaning once, rather than repeated under every tool that takes it.
    """
    lines, used = [], []
    for name in registry():
        req = required_tool_kwargs(name)
        lines.append(f"- {name}: " + (f"tool_kwargs MUST include {req}" if req else "no tool_kwargs needed"))
        used += [a for a in req if a not in used]
    lines.append("\nWhat each argument means:")
    lines += [f"- {a}: {ARG_HELP.get(a, 'required')}" for a in used]
    return "\n".join(lines)


def normalize_tool_kwargs(
    tool_name: str, kwargs: dict, n_prompts: int
) -> tuple[dict, list[str], list[str]]:
    """Coerce LLM-produced tool_kwargs into what the tool actually accepts.

    Renames known aliases, broadcasts a single token pair across all prompts
    (ablation and activation_patching zip tokens against prompts, so a short
    list would silently measure only the first few), and drops arguments the
    tool doesn't take.

    Args:
        tool_name: Key into the active backend's registry().
        kwargs: The spec's tool_kwargs, as parsed from the plan.
        n_prompts: How many prompts the experiment runs on.

    Returns:
        (cleaned kwargs, required arguments still missing, arguments dropped).
        Dropped names are reported so an unsupported request — asking ablation
        to target specific heads, say — is visible rather than silent.
    """
    accepted = set(inspect.signature(registry()[tool_name]).parameters)
    out: dict = {}
    dropped: list[str] = []
    for key, value in (kwargs or {}).items():
        canonical = KWARG_ALIASES.get(key, key)
        if canonical in _PAIR_ARGS and canonical not in accepted:
            canonical = next((a for a in _PAIR_ARGS if a in accepted), canonical)
        if canonical in accepted and canonical not in _NODE_SUPPLIED:
            out[canonical] = value
        else:
            dropped.append(key)

    for arg in _BROADCAST_ARGS:
        value = out.get(arg)
        if isinstance(value, str):
            value = [value]
        if isinstance(value, list) and len(value) == 1 and n_prompts > 1:
            value = value * n_prompts          # one pair, applied to every prompt
        if value is not None:
            out[arg] = value

    missing = [a for a in required_tool_kwargs(tool_name) if a not in out]
    return out, missing, dropped


def token_defaults_from_specs(specs: list[dict]) -> dict:
    """Pull the question-level token pair out of a parsed plan.

    positive_tokens/negative_tokens describe the behaviour under study, not
    one experiment, so the first spec that names them supplies the default
    for follow-ups whose own kwargs omit them.
    """
    for spec in specs:
        kwargs = {
            KWARG_ALIASES.get(k, k): v for k, v in (spec.get("tool_kwargs") or {}).items()
        }
        if all(a in kwargs for a in _TOKEN_ARGS):
            return {a: kwargs[a] for a in _TOKEN_ARGS}
    return {}


def prompt_pairs_from_specs(specs: list[dict]) -> dict[str, list[str]]:
    """Map each planned prompt set to its swapped counterpart, keyed by the prompts joined with newlines.

    A follow-up reruns the previous experiment's prompts, and the planning LLM
    that writes it never sees their corrupted/counterfactual partners — so
    they're remembered here and filled in when the follow-up's tool needs them.
    """
    pairs: dict[str, list[str]] = {}
    for spec in specs:
        kwargs = {KWARG_ALIASES.get(k, k): v for k, v in (spec.get("tool_kwargs") or {}).items()}
        for arg in _PAIR_ARGS:
            if kwargs.get(arg):
                pairs.setdefault("\n".join(spec["prompts"]), kwargs[arg])
    return pairs


__all__ = [
    "run_logit_lens",
    "run_attention_pattern",
    "run_direct_logit_attribution",
    "run_ablation",
    "run_activation_patching",
    "run_attribution_patching",
    "run_head_patching",
    "run_circuit_evaluation",
    "run_counterfactual",
    "run_linear_probe",
    "run_steering_vector",
    "run_directional_ablation",
    "run_feature_search",
    "run_feature_attribution",
    "run_feature_clamping",
    "run_edge_attribution_patching",
    "run_path_patching",
    "run_circuit_tracing",
    "TOOL_REGISTRY",
    "SEESAW_REGISTRY",
    "LEVER_REGISTRY",
    "registry",
    "ARG_HELP",
    "KWARG_ALIASES",
    "required_tool_kwargs",
    "tool_kwargs_guide",
    "normalize_tool_kwargs",
    "token_defaults_from_specs",
    "prompt_pairs_from_specs",
]
