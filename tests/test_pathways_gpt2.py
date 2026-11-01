"""Known-answer tests: each pathway on GPT-2 small must recover what's already established.

- circuit_discovery on IOI must find the circuit of Wang et al. (2022): the
  S-inhibition heads, L9H9 as a name mover, L10H7/L11H10 as negative name
  movers, and a faithful circuit. Its nominating step (attribution
  patching) must rank heads the way real patching does.
- attribute_audit must find the occupation→pronoun bias (Vig et al. 2020)
  and must *not* find one where there isn't any.
- feature_explain must find SAE features that, switched off, remove most of
  the IOI behaviour — far more than random features.

This is what makes a pathway a baseline Lens can be scored against: it's
checked against the literature, not just run.

Needs GPT-2 small and one GPT-2 SAE (~150 MB) and network for Neuronpedia
labels (failures there leave labels empty, they don't fail the test), so
it's opt-in: SEESAW_MODEL_TESTS=1 pytest tests/test_pathways_gpt2.py
"""

from __future__ import annotations

import os

import pytest

if not os.environ.get("SEESAW_MODEL_TESTS"):
    pytest.skip("set SEESAW_MODEL_TESTS=1 to run GPT-2 known-answer tests", allow_module_level=True)

torch = pytest.importorskip("torch")
pytest.importorskip("transformer_lens")

from transformer_lens import HookedTransformer  # noqa: E402

from seesaw.interp.causal_interventions import attribution_patching, head_patching  # noqa: E402
from seesaw.interp.circuit_finding import (  # noqa: E402
    circuit_tracing, edge_attribution_patching, path_patching,
)
from seesaw.interp.feature_extraction.sae import load_sae  # noqa: E402
from seesaw.pathways import attribute_audit, circuit_discovery, feature_explain  # noqa: E402

NAMES = [("Mary", "John"), ("Alice", "Bob"), ("Sarah", "Tom"), ("Anna", "James"),
         ("Kate", "Mark"), ("Lisa", "David"), ("Emma", "Paul"), ("Rachel", "Steve")]
IOI       = [f"When {a} and {b} went to the store, {b} gave a drink to" for a, b in NAMES]
IOI_CORR  = [f"When {a} and {b} went to the store, {a} gave a drink to" for a, b in NAMES]
IOI_ABC   = [f"When {a} and {b} went to the store, Mike gave a drink to" for a, b in NAMES]
IO, S     = [f" {a}" for a, _ in NAMES], [f" {b}" for _, b in NAMES]

S_INHIBITION = {"L7H3", "L7H9", "L8H6", "L8H10"}
NEGATIVE_NAME_MOVERS = {"L10H7", "L11H10"}

FEMALE = ["nurse", "secretary", "nanny", "librarian", "dancer", "maid", "teacher", "cashier"]
MALE   = ["engineer", "mechanic", "carpenter", "plumber", "pilot", "farmer", "soldier", "driver"]


@pytest.fixture(scope="module")
def gpt2():
    return HookedTransformer.from_pretrained("gpt2")


@pytest.fixture(scope="module")
def ioi(gpt2, tmp_path_factory):
    return circuit_discovery.run(
        gpt2, IOI, IO, S, tmp_path_factory.mktemp("cd"),
        corrupted_prompts=IOI_CORR, reference_prompts=IOI_ABC)


def test_ioi_circuit_is_found(ioi):
    assert ioi.verdict == "circuit_found", ioi.to_markdown()


def test_ioi_finds_s_inhibition_heads_upstream(ioi):
    found = S_INHIBITION & set(ioi.findings["necessary_but_indirect"])
    assert len(found) >= 3, ioi.findings["necessary_but_indirect"]


def test_ioi_finds_name_movers(ioi):
    assert "L9H9" in ioi.findings["confirmed_writers"]
    writers = set(ioi.findings["confirmed_writers"]) | set(ioi.findings["writes_but_redundant"])
    assert "L9H6" in writers


def test_ioi_finds_negative_name_movers(ioi):
    assert NEGATIVE_NAME_MOVERS <= set(ioi.findings["writes_against_answer"])


def test_attribution_patching_ranks_heads_like_real_patching(gpt2, tmp_path):
    atp = attribution_patching.compute(gpt2, IOI, IOI_CORR, IO, S, tmp_path)
    heads = [f"L{l}H{h}" for l in range(12) for h in range(12)]
    real = head_patching.compute(gpt2, IOI, IOI_CORR, IO, S, heads, tmp_path)
    a = torch.tensor([atp.data["head_effects"][l][h] for l in range(12) for h in range(12)])
    b = torch.tensor([real.data["per_head"][n] for n in heads])
    rank = lambda x: x.argsort().argsort().float()  # noqa: E731
    spearman = torch.corrcoef(torch.stack([rank(a), rank(b)]))[0, 1].item()
    assert spearman > 0.8, spearman


def test_ioi_wiring_includes_induction_to_s_inhibition(ioi):
    """Wang et al.: induction heads feed the S-inhibition heads through their values."""
    edges = ioi.findings["confirmed_edges"]
    assert "L5H5->L8H6.v" in edges, sorted(edges)
    assert edges["L9H9->logits"] > 0 and edges["L10H7->logits"] < 0


def test_eap_ig_estimates_match_path_patching(gpt2, tmp_path):
    eap = edge_attribution_patching.compute(gpt2, IOI, IOI_CORR, IO, S, tmp_path, top_k=30, ig_steps=5)
    edges = [e for e, _ in eap.data["top_edges"]]
    real = path_patching.compute(gpt2, IOI, IOI_CORR, IO, S, edges, tmp_path)
    est = torch.tensor([s for _, s in eap.data["top_edges"]])
    got = torch.tensor([real.data["per_edge"][e] for e in edges])
    assert torch.corrcoef(torch.stack([est, got]))[0, 1].item() > 0.9


def test_circuit_tracing_finds_texas_between_dallas_and_austin(tmp_path):
    """Ameisen et al. 2025: Dallas → Texas → Austin, a two-hop path in Gemma-2-2B.
    Reads the public graph Neuronpedia hosts; generates nothing."""
    r = circuit_tracing.compute(tmp_path, slug="gemma-fact-dallas-austin", top_k=15)
    assert "Austin" in r.data["target"]
    labels = [(f["token"], (f["explanation"] or "").lower()) for f in r.data["top_features"]]
    assert any(tok == " Dallas" and "texas" in label for tok, label in labels), labels


def test_gender_bias_is_found(gpt2, tmp_path):
    r = attribute_audit.run(
        gpt2, [f"The {o} said that" for o in FEMALE], [f"The {o} said that" for o in MALE],
        [" she"] * 8, [" he"] * 8, tmp_path,
        control_texts=["The weather today is mild and the park was full of people walking their dogs."])
    assert r.verdict == "uses_attribute", r.to_markdown()


def test_irrelevant_attribute_is_not_flagged(gpt2, tmp_path):
    foods_a = ["apple", "banana", "orange", "grape"]
    foods_b = ["sandwich", "salad", "pizza", "burger"]
    r = attribute_audit.run(
        gpt2, [f"The man who ate the {f} said that" for f in foods_a],
        [f"The man who ate the {f} said that" for f in foods_b],
        [" she"] * 4, [" he"] * 4, tmp_path)
    assert r.verdict == "no_effect", r.to_markdown()


def test_sae_features_explain_ioi(gpt2, tmp_path):
    sae = load_sae("gpt2-small-res-jb", "blocks.8.hook_resid_pre")
    r = feature_explain.run(gpt2, sae, IOI, IO, S, tmp_path, search_neuronpedia=False)
    assert r.verdict == "features_explain", r.to_markdown()
