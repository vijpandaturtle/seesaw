"""The evidence contract: every tool has an entry, and assessments say what's missing.

No torch: this is what Lens, Quill and the pathways read to label results.
tests/test_lens_tools.py checks the control keys named here are ones the
tools really return.
"""

from __future__ import annotations

import pytest

from shared.evidence import TRUST, as_text, assess, refusal
from shared.lens_tools import BACKENDS, catalog


@pytest.mark.parametrize("backend", BACKENDS)
def test_every_catalogued_tool_has_a_trust_entry(backend):
    assert set(TRUST[backend]) == set(catalog(backend))


def test_kind_sets_the_plain_strength():
    assert assess("logit_lens", {}, backend="seesaw")["strength"] == "Correlational"
    assert assess("attribution_patching", {}, backend="seesaw")["strength"] == "Estimate"
    assert assess("head_patching", {}, backend="seesaw")["strength"] == "Causal"


def test_a_missing_required_control_makes_the_result_incomplete():
    a = assess("directional_ablation", {"fraction_removed": 0.9, "random_fraction_removed": None},
               n_prompts=16, backend="seesaw")
    assert not a["complete"]
    assert "incomplete" in a["label"] and "random directions" in a["label"]
    assert "MISSING" in as_text(a)


def test_all_controls_present_is_labelled_controlled():
    a = assess("directional_ablation", {"random_fraction_removed": 0.05, "control_loss_after": 3.1},
               n_prompts=16, backend="seesaw")
    assert a["complete"] and a["label"] == "Causal, controlled"


def test_optional_controls_never_make_a_result_incomplete():
    a = assess("counterfactual", {"control_mean_ld": None}, n_prompts=16, backend="seesaw")
    assert a["complete"] and a["label"] == "Correlational"
    assert "not run (optional)" in as_text(a)


def test_few_prompts_are_flagged_but_single_example_tools_are_not():
    thin = assess("head_patching", {}, n_prompts=3, backend="seesaw")
    assert "3 prompts" in thin["label"]
    assert any("not for the behaviour in general" in c for c in thin["caveats"])
    assert "prompt" not in assess("circuit_tracing", {}, n_prompts=1, backend="seesaw")["label"]


def test_a_probe_refuses_too_few_prompts():
    assert refusal("linear_probe", 3, backend="seesaw")
    assert refusal("linear_probe", 4, backend="seesaw") is None
    assert refusal("head_patching", 1, backend="seesaw") is None
