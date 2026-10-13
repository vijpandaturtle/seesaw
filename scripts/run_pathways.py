"""Run every pathway on its known-answer task in GPT-2 small and save the reports.

    python scripts/run_pathways.py                 # all three
    python scripts/run_pathways.py attribute_audit # just one

Writes outputs/pathways/<case>/report.md (the pathway's one-page verdict and
evidence) next to every plot its steps saved. The same cases are asserted in
tests/test_pathways_gpt2.py; this script is for looking at them.

Needs the [experiments] extra; feature_explain downloads one GPT-2 SAE
(~150 MB) and calls Neuronpedia for feature labels.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "outputs" / "pathways"

NAMES = [("Mary", "John"), ("Alice", "Bob"), ("Sarah", "Tom"), ("Anna", "James"),
         ("Kate", "Mark"), ("Lisa", "David"), ("Emma", "Paul"), ("Rachel", "Steve")]
IOI      = [f"When {a} and {b} went to the store, {b} gave a drink to" for a, b in NAMES]
IOI_CORR = [f"When {a} and {b} went to the store, {a} gave a drink to" for a, b in NAMES]
IOI_ABC  = [f"When {a} and {b} went to the store, Mike gave a drink to" for a, b in NAMES]
IO, S    = [f" {a}" for a, _ in NAMES], [f" {b}" for _, b in NAMES]

FEMALE = ["nurse", "secretary", "nanny", "librarian", "dancer", "maid", "teacher", "cashier"]
MALE   = ["engineer", "mechanic", "carpenter", "plumber", "pilot", "farmer", "soldier", "driver"]
CONTROL_TEXTS = [
    "The weather today is mild and the park was full of people walking their dogs.",
    "Stock prices rose sharply after the central bank announced its decision.",
]


def circuit_discovery(model):
    from seesaw.pathways import circuit_discovery as pw

    return pw.run(model, IOI, IO, S, OUT / "circuit_discovery",
                  corrupted_prompts=IOI_CORR, reference_prompts=IOI_ABC)


def attribute_audit(model):
    from seesaw.pathways import attribute_audit as pw

    return pw.run(model, [f"The {o} said that" for o in FEMALE], [f"The {o} said that" for o in MALE],
                  [" she"] * len(FEMALE), [" he"] * len(FEMALE), OUT / "attribute_audit",
                  control_prompts=["The person said that", "The student said that", "The worker said that"],
                  control_texts=CONTROL_TEXTS)


def attribute_audit_control(model):
    """The negative control: food has nothing to do with pronouns."""
    from seesaw.pathways import attribute_audit as pw

    a, b = ["apple", "banana", "orange", "grape"], ["sandwich", "salad", "pizza", "burger"]
    return pw.run(model, [f"The man who ate the {f} said that" for f in a],
                  [f"The man who ate the {f} said that" for f in b],
                  [" she"] * 4, [" he"] * 4, OUT / "attribute_audit_control")


def feature_explain(model):
    from seesaw.interp.feature_extraction.sae import load_sae
    from seesaw.pathways import feature_explain as pw

    sae = load_sae("gpt2-small-res-jb", "blocks.8.hook_resid_pre")
    return pw.run(model, sae, IOI, IO, S, OUT / "feature_explain")


CASES = {f.__name__: f for f in (circuit_discovery, attribute_audit, attribute_audit_control, feature_explain)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cases", nargs="*", choices=[[], *CASES], help="default: all")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", module="sae_lens")

    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained("gpt2")
    for name in args.cases or CASES:
        result = CASES[name](model)
        (OUT / name).mkdir(parents=True, exist_ok=True)
        report = OUT / name / "report.md"
        report.write_text(result.to_markdown())
        print(f"{name:<26} {result.verdict:<20} {report.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
