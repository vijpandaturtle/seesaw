import inspect

from .logit_lens import run_logit_lens
from .attention_pattern import run_attention_pattern
from .direct_logit_attribution import run_direct_logit_attribution
from .ablation import run_ablation
from .activation_patching import run_activation_patching

TOOL_REGISTRY: dict = {
    "logit_lens":               run_logit_lens,
    "attention_pattern":        run_attention_pattern,
    "direct_logit_attribution": run_direct_logit_attribution,
    "ablation":                 run_ablation,
    "activation_patching":      run_activation_patching,
    # Tier 2 (reasoning models): cot_faithfulness, thinking_token_lens, thought_intervention
    # Tier 3 (SAE): sae_feature_search, feature_steering
    # Tier 4 (safety): linear_probe, representation_reading
}

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
}

_TOKEN_ARGS = ("positive_tokens", "negative_tokens")


def required_tool_kwargs(tool_name: str) -> list[str]:
    """Extra arguments a tool needs beyond model and prompts.

    Read off the live signature, so adding a tool or renaming a parameter
    updates both the parse_plan prompt and dispatch validation.
    """
    fn = TOOL_REGISTRY[tool_name]
    return [
        p.name
        for p in inspect.signature(fn).parameters.values()
        if p.default is inspect.Parameter.empty and p.name not in _NODE_SUPPLIED
    ]


def tool_kwargs_guide() -> str:
    """Per-tool tool_kwargs contract, formatted for the parse_plan prompt."""
    lines = []
    for name in TOOL_REGISTRY:
        req = required_tool_kwargs(name)
        if not req:
            lines.append(f"- {name}: no tool_kwargs needed")
            continue
        args = "; ".join(f"{a} — {ARG_HELP.get(a, 'required')}" for a in req)
        lines.append(f"- {name}: tool_kwargs MUST include {req} where {args}")
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
        tool_name: Key into TOOL_REGISTRY.
        kwargs: The spec's tool_kwargs, as parsed from the plan.
        n_prompts: How many prompts the experiment runs on.

    Returns:
        (cleaned kwargs, required arguments still missing, arguments dropped).
        Dropped names are reported so an unsupported request — asking ablation
        to target specific heads, say — is visible rather than silent.
    """
    accepted = set(inspect.signature(TOOL_REGISTRY[tool_name]).parameters)
    out: dict = {}
    dropped: list[str] = []
    for key, value in (kwargs or {}).items():
        canonical = KWARG_ALIASES.get(key, key)
        if canonical in accepted and canonical not in _NODE_SUPPLIED:
            out[canonical] = value
        else:
            dropped.append(key)

    for arg in _TOKEN_ARGS:
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


__all__ = [
    "run_logit_lens",
    "run_attention_pattern",
    "run_direct_logit_attribution",
    "run_ablation",
    "run_activation_patching",
    "TOOL_REGISTRY",
    "ARG_HELP",
    "KWARG_ALIASES",
    "required_tool_kwargs",
    "tool_kwargs_guide",
    "normalize_tool_kwargs",
    "token_defaults_from_specs",
]
