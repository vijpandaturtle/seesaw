"""What a job investigates: a behavior, shown with examples, not a research question.

A developer rarely arrives with a research question. They arrive with a bad
output they saw and something they need to do about it. So a job starts from:

- the behavior, in a sentence ("gives up a correct answer when the user
  pushes back"). It's stored as the job's `question` text, which is what the
  job list and the agents show;
- examples: real prompts where the model does it, with what it said;
- non-examples: the closest prompts where it doesn't;
- the action: what to do about it. One of the use-case areas
  (seesaw.use_cases.AREAS): monitor it at runtime, audit where it comes from,
  fix it, or understand how the model does it.

Only the behavior is required. The examples pin down what it means far better
than any description, and give Scout something to vary instead of inventing
prompts from nothing, which is where generated data goes formulaic. The
action narrows the methods to that area's use cases and decides what the
result is scored on: a detector against text-only baselines, a fix against
prompting, and so on.

The rest is stored as params["behavior"]:

    {"examples": [{"prompt": "...", "output": "..."}, ...],
     "non_examples": [...], "action": "monitor" | "audit" | "fix" | "understand"}

`clean` checks it, `load` reads it from a YAML or JSON file, and `brief`
turns it into the section of Scout's brief that says what to investigate.
"""

from __future__ import annotations

import json
from pathlib import Path

KEYS = ("examples", "non_examples", "action")
# What the user can do about a behavior: the use-case areas, in display order.
ACTIONS = ("monitor", "audit", "fix", "understand")
# Long enough for a real conversation, short enough to keep Scout's brief readable.
MAX_TEXT = 4000
MAX_EXAMPLES = 20


def _case(item, where: str) -> dict:
    if isinstance(item, str):
        item = {"prompt": item}
    if not isinstance(item, dict) or not set(item) <= {"prompt", "output"}:
        raise ValueError(f"{where}: each example is a prompt, or {{\"prompt\": ..., \"output\": ...}}")
    prompt = item.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{where}: an example needs a non-empty prompt")
    case = {"prompt": prompt.strip()}
    output = item.get("output")
    if output is not None:
        if not isinstance(output, str):
            raise ValueError(f"{where}: an example's output is text")
        if output.strip():
            case["output"] = output.strip()
    for key, text in case.items():
        if len(text) > MAX_TEXT:
            raise ValueError(f"{where}: an example's {key} is over {MAX_TEXT} characters")
    return case


def clean(behavior: dict | None) -> dict:
    """params["behavior"] checked and normalised; empty parts are dropped.

    Raises:
        ValueError: unknown keys, an example without a prompt, too many
            examples, or text that is too long.
    """
    if not behavior:
        return {}
    if not isinstance(behavior, dict) or not set(behavior) <= set(KEYS):
        raise ValueError(f"behavior takes only {list(KEYS)}")
    out: dict = {}
    for key in ("examples", "non_examples"):
        items = behavior.get(key) or []
        if not isinstance(items, list):
            raise ValueError(f"behavior.{key} is a list")
        if len(items) > MAX_EXAMPLES:
            raise ValueError(f"behavior.{key}: at most {MAX_EXAMPLES}")
        cases = [_case(item, f"behavior.{key}") for item in items]
        if cases:
            out[key] = cases
    action = behavior.get("action")
    if action is not None and action != "":
        if action not in ACTIONS:
            raise ValueError(f"behavior.action is one of {list(ACTIONS)}, not {action!r}")
        out["action"] = action
    return out


def load(path: str | Path) -> tuple[str, dict]:
    """Read a behavior file: YAML (or JSON) with `behavior` and any of examples, non_examples, action.

    Returns:
        (the behavior sentence, the cleaned params["behavior"]).
    """
    text = Path(path).read_text()
    if Path(path).suffix == ".json":
        data = json.loads(text)
    else:
        import yaml

        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping with a `behavior` sentence")
    data = dict(data)
    sentence = data.pop("behavior", None)
    if not isinstance(sentence, str) or not sentence.strip():
        raise ValueError(f"{path}: `behavior` is required, one sentence saying what the model does")
    return sentence.strip(), clean(data)


def brief(sentence: str, behavior: dict | None) -> str:
    """The part of Scout's brief that says what to investigate."""
    behavior = behavior or {}
    lines = [
        "## The behavior to investigate",
        "",
        f"The user wants to investigate this behavior: {sentence.strip()}",
        "",
        "Before choosing experiments, restate it in the plan as one testable claim about this",
        "model (what it does, on which prompts, compared with what), and say what result would",
        "support it and what would count against it.",
    ]

    def show(cases: list[dict]) -> list[str]:
        out = []
        for i, case in enumerate(cases, 1):
            out.append(f"{i}. Prompt: {json.dumps(case['prompt'])}")
            if "output" in case:
                out.append(f"   Model said: {json.dumps(case['output'])}")
        return out

    if behavior.get("examples"):
        lines += ["", "Real cases where the model does it, from the user:", "",
                  *show(behavior["examples"]), "",
                  "Build the prompts from these: vary them (topic, wording, length) rather than",
                  "inventing a different kind of prompt, and keep them as close to these as the",
                  "inputs allow."]
    else:
        lines += ["", "The user gave no examples. Write prompts where the behavior should show, and",
                  "say in the plan that they are yours, not the user's."]
    if behavior.get("non_examples"):
        lines += ["", "The closest cases where the model doesn't do it:", "",
                  *show(behavior["non_examples"]), "",
                  "Use these as the pattern for contrast prompts: the same kind of prompt without",
                  "the trigger, not a different topic. Match lengths, so text length alone can't",
                  "tell the two apart."]
    if behavior.get("action"):
        lines += ["", *_action_brief(behavior["action"])]
    return "\n".join(lines)


def _action_brief(action: str) -> list[str]:
    """What the user wants done, and the use cases (with their pathways) that do it."""
    from seesaw.use_cases import AREAS, USE_CASES

    lines = [f"What the user wants to do about it: {AREAS[action]}.", ""]
    cases = [(name, u) for name, u in USE_CASES.items() if u.area == action and u.status != "planned"]
    if cases:
        lines.append("The use cases for that, with the pathways that do them:")
        for name, u in cases:
            lines.append(f"- {u.title} (`{name}`): {', '.join(f'`{p}`' for p in u.pathways)}. "
                         + (f"Scored by {u.metric}; compared with {u.baseline}."
                            if action != "understand" else f"Reports {u.metric}."))
        lines.append("")
    if action == "understand":
        lines.append("There is no outcome to beat: explain how the model does it.")
    else:
        lines.append("Aim the plan at that: say how the result will be scored, and include the "
                     "baseline without interpretability it has to beat.")
    return lines
