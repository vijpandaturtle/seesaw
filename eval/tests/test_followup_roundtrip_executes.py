"""Follow-ups: Quill's suggestions must be executable by Lens, not just prose.

Quill emits follow-up experiments naming a tool and its parameters, and Lens
picks them up through the same dispatch path as planned experiments. That
round trip has broken twice in this repo — both times because the model was
told about tools without being told their argument contract, so it invented
parameters (`io_tokens`, then `heads`/`layers`) that no tool accepts and every
generated follow-up failed before running.

These checks assert the contract holds: whatever the run produced, follow-ups
name real tools and actually execute.
"""

import json
import os
from pathlib import Path

import pytest

from conftest import require_tool

LENS_TOOLS = {
    "logit_lens",
    "attention_pattern",
    "direct_logit_attribution",
    "ablation",
    "activation_patching",
}


@pytest.fixture(scope="session")
def report() -> dict:
    raw = os.environ.get("SEESAW_EVAL_REPORT")
    if not raw or not Path(raw).exists():
        pytest.fail("no critique report for this run — set SEESAW_EVAL_REPORT")
    path = Path(raw)
    # save_critique writes .json alongside .md; the JSON is the parseable one.
    if path.suffix == ".md":
        sibling = path.with_suffix(".json")
        if sibling.exists():
            path = sibling
    if path.suffix != ".json":
        pytest.fail(f"critique report at {path} is not JSON, so it can't be checked")
    return json.loads(path.read_text())


def test_followups_name_real_tools(report):
    followups = report.get("followups") or []
    for f in followups:
        tool = f.get("tool")
        assert tool in LENS_TOOLS, (
            f"follow-up {f.get('name', '?')!r} names tool {tool!r}, which is "
            f"not in the registry {sorted(LENS_TOOLS)} — Lens cannot run it"
        )


def test_a_followup_actually_ran(bundle, by_tool):
    """The bundle records follow-ups as results named 'Follow-up: ...'. If the
    run produced any, at least one has to have succeeded — a follow-up that
    always fails is the exact regression this file exists to catch."""
    followup_results = [
        r for r in bundle.get("results", [])
        if str(r.get("name", "")).startswith("Follow-up")
    ]
    if not followup_results:
        return          # Lens is conservative about follow-ups; none is valid

    succeeded = [r for r in followup_results if r.get("status") == "success"]
    assert succeeded, (
        "every follow-up experiment failed: "
        + "; ".join(
            f"{r.get('tool')}: {(r.get('error') or '')[:120]}"
            for r in followup_results
        )
    )


def test_followup_results_are_attributed(bundle):
    """Failed results used to record tool='unknown', which made it impossible
    to tell which tool broke."""
    for r in bundle.get("results", []):
        if str(r.get("name", "")).startswith("Follow-up"):
            assert r.get("tool") in LENS_TOOLS, (
                f"follow-up result records tool={r.get('tool')!r}; failures "
                f"must still name the tool that produced them"
            )
