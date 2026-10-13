"""Lens's tool wiring: the catalog agrees with the registry, and every tool runs through dispatch.

shared/lens_tools.py is what Scout, Quill and the graders read; TOOL_REGISTRY
is what Lens runs. If they disagree, Scout plans experiments Lens can't run
or the graders reject ones it can — so they're checked against each other.

The dispatch test sends each tool through normalize_tool_kwargs and
run_in_sandbox exactly as run_experiment does, on a tiny random model, so a
wrapper whose signature, kwargs contract or result shape is wrong fails here
rather than mid-pipeline. The SAE tools and circuit_tracing are left out
(they need a published SAE, or Neuronpedia's graph service);
tests/test_pathways_gpt2.py exercises them.
"""

from __future__ import annotations

import json

import pytest

from shared.lens_tools import LENS_TOOLS, names


def test_catalog_kinds_are_known():
    assert {t.kind for t in LENS_TOOLS.values()} <= {"descriptive", "estimate", "causal"}
    assert names("causal") and names("estimate") and names("descriptive")


torch = pytest.importorskip("torch")
pytest.importorskip("transformer_lens")

from transformer_lens import HookedTransformer, HookedTransformerConfig  # noqa: E402

from lens.mcp_server.src.app.sandbox import run_in_sandbox  # noqa: E402
from lens.mcp_server.src.tools import (  # noqa: E402
    TOOL_REGISTRY, normalize_tool_kwargs, prompt_pairs_from_specs,
)

PROMPTS   = ["When Mary and John went to the store, John gave a drink to",
             "When Alice and Bob went to the park, Bob handed a book to"]
CORRUPTED = ["When Mary and John went to the store, Mary gave a drink to",
             "When Alice and Bob went to the park, Alice handed a book to"]

# What a plan would supply, by argument name. Deliberately one shared pool:
# normalize_tool_kwargs must drop what a tool doesn't take.
PLAN_KWARGS = {
    "positive_tokens":   [" Mary"],         # one pair, broadcast across prompts
    "negative_tokens":   [" John"],
    "corrupted_prompts": CORRUPTED,         # routed to counterfactual_prompts where needed
    "heads":             ["L1H2", "L0H1"],
    "edges":             ["L0H1->L1H2.q", "L1H2->logits"],
}
# Need a published SAE for the model, or Neuronpedia's graph service.
EXTERNAL_TOOLS = {"feature_search", "feature_attribution", "feature_clamping", "circuit_tracing"}


def test_catalog_matches_registry():
    assert list(LENS_TOOLS) == list(TOOL_REGISTRY)


@pytest.fixture(scope="module")
def model():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=32, d_head=8, n_heads=4, d_mlp=64, n_ctx=64,
        act_fn="gelu", tokenizer_name="gpt2", normalization_type="LN", seed=0)
    return HookedTransformer(cfg).eval()


@pytest.mark.parametrize("tool", [t for t in LENS_TOOLS if t not in EXTERNAL_TOOLS])
def test_every_tool_runs_through_dispatch(tool, model, tmp_path):
    kwargs, missing, _ = normalize_tool_kwargs(tool, PLAN_KWARGS, len(PROMPTS))
    assert not missing, f"{tool} needs {missing}, which a plan using the documented names would provide"
    result = run_in_sandbox(
        TOOL_REGISTRY[tool],
        tool_kwargs={"model": model, "prompts": PROMPTS, "output_dir": tmp_path, **kwargs},
        experiment_name=tool, tool=tool,
    )
    assert result.status == "success", result.error
    assert result.tool == tool
    json.dumps(result.data)                                  # goes into the bundle as JSON
    assert all(p.exists() for p in result.plot_paths)


def test_pair_prompts_route_to_whichever_name_the_tool_takes():
    kwargs, _, dropped = normalize_tool_kwargs("linear_probe", {"corrupted_prompts": CORRUPTED}, 2)
    assert kwargs == {"counterfactual_prompts": CORRUPTED} and not dropped
    kwargs, _, _ = normalize_tool_kwargs("head_patching", {"contrast_prompts": CORRUPTED}, 2)
    assert kwargs["corrupted_prompts"] == CORRUPTED


def test_follow_ups_can_recover_a_plans_prompt_pairs():
    specs = [
        {"prompts": PROMPTS, "tool_kwargs": {"corrupt_prompts": CORRUPTED}},
        {"prompts": ["The nurse said that"], "tool_kwargs": {"counterfactual_prompts": ["The pilot said that"]}},
        {"prompts": ["x"], "tool_kwargs": {}},
    ]
    pairs = prompt_pairs_from_specs(specs)
    assert pairs == {"\n".join(PROMPTS): CORRUPTED, "The nurse said that": ["The pilot said that"]}
