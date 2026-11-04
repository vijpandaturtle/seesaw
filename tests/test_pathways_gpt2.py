"""Known-answer tests: each pathway on GPT-2 small must recover what's already established.

- circuit_discovery on IOI must find the circuit of Wang et al. (2022): the
  S-inhibition heads, L9H9 as a name mover, L10H7/L11H10 as negative name
  movers, and a faithful circuit. Its nominating step (attribution
  patching) must rank heads the way real patching does.
- concept_direction must find the occupation→pronoun bias (Vig et al. 2020)
  on held-out pairs, steer it, and bake the removal into the weights; and
  must *not* find one where there isn't any.
- feature_explain must find SAE features that, switched off, remove most of
  the IOI behaviour — far more than random features.
- probe_monitor must call an obvious-wording sentiment probe lexical_only:
  the words alone carry it.
- spurious_audit must find no reliance of sentiment on reviewers' names, and
  must find (and erase, far beyond random erasers) gendered names' pull on
  an occupation the context states.
- factual_recall must show the residual-stream hand-off of Meng et al. 2022:
  the subject's last token carries the fact early, the final token late.
- editing_comparison: ROME edits work at middle layers and can't at the last.
- model_diffing must locate a change planted in one block of a copy of GPT-2.

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
from seesaw.pathways import (  # noqa: E402
    attribute_audit, circuit_discovery, concept_direction, editing_comparison, factual_recall,
    feature_explain, model_diffing, probe_monitor, spurious_audit,
)

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


CONTROL_TEXT = ["The weather today is mild and the park was full of people walking their dogs."]


def test_gender_bias_is_found_on_held_out_pairs(gpt2, tmp_path):
    r = concept_direction.run(
        gpt2, [f"The {o} said that" for o in FEMALE], [f"The {o} said that" for o in MALE],
        [" she"] * 8, [" he"] * 8, tmp_path, control_texts=CONTROL_TEXT)
    assert r.verdict in {"uses_direction", "partly_uses_direction"}, r.to_markdown()
    # Held out, removal beats random directions but doesn't close the whole gap:
    # in-sample, the old attribute_audit called this "uses_attribute".
    assert r.findings["fraction_removed"] - r.findings["random_fraction_removed"] >= 0.1
    assert r.findings["steerable"] and r.findings["bakeable"], r.to_markdown()


def test_attribute_audit_is_concept_direction_under_its_old_name(gpt2, tmp_path):
    r = attribute_audit.run(
        gpt2, [f"The {o} said that" for o in FEMALE], [f"The {o} said that" for o in MALE],
        [" she"] * 8, [" he"] * 8, tmp_path, control_texts=CONTROL_TEXT)
    assert r.pathway == "attribute_audit" and r.verdict in {"uses_direction", "partly_uses_direction"}


def test_irrelevant_attribute_is_not_flagged(gpt2, tmp_path):
    foods_a = ["apple", "banana", "orange", "grape"]
    foods_b = ["sandwich", "salad", "pizza", "burger"]
    r = concept_direction.run(
        gpt2, [f"The man who ate the {f} said that" for f in foods_a],
        [f"The man who ate the {f} said that" for f in foods_b],
        [" she"] * 4, [" he"] * 4, tmp_path)
    assert r.verdict == "no_effect", r.to_markdown()


def test_sae_features_explain_ioi(gpt2, tmp_path):
    sae = load_sae("gpt2-small-res-jb", "blocks.8.hook_resid_pre")
    r = feature_explain.run(gpt2, sae, IOI, IO, S, tmp_path, search_neuronpedia=False)
    assert r.verdict == "features_explain", r.to_markdown()


def test_sae_layer_sweep_reports_each_layer(gpt2, tmp_path):
    sae = load_sae("gpt2-small-res-jb", "blocks.8.hook_resid_pre")
    r = feature_explain.run(gpt2, [sae], IOI, IO, S, tmp_path, search_neuronpedia=False)
    assert r.verdict == "features_explain" and r.findings["layer"] == "blocks.8.hook_resid_pre"


SENT_POS = ["This movie was wonderful and I loved every minute.", "A brilliant film with a moving story.",
            "The acting was superb and the ending was perfect.", "I really enjoyed this delightful picture.",
            "One of the best films I have seen in years.", "A charming, funny and beautiful movie.",
            "The director did an amazing job, truly great.", "I was thrilled by this fantastic film."]
SENT_NEG = ["This movie was awful and I hated every minute.", "A dreadful film with a boring story.",
            "The acting was terrible and the ending was stupid.", "I really disliked this dull picture.",
            "One of the worst films I have seen in years.", "A tedious, silly and ugly movie.",
            "The director did a horrible job, truly bad.", "I was bored by this miserable film."]
TEST_POS = ["The blender works great and I love it.", "Excellent phone, fast and reliable.",
            "These shoes are comfortable and well made.", "Fantastic service, the staff were lovely.",
            "The hotel room was clean and the view was gorgeous.", "My new laptop is a joy to use."]
TEST_NEG = ["The blender broke and I regret buying it.", "Terrible phone, slow and unreliable.",
            "These shoes are painful and badly made.", "Awful service, the staff were rude.",
            "The hotel room was dirty and the view was depressing.", "My new laptop is a pain to use."]


def test_obvious_sentiment_is_flagged_lexical(gpt2, tmp_path):
    """It transfers to product reviews, but the words alone do as well: not worth the model."""
    r = probe_monitor.run(gpt2, SENT_POS, SENT_NEG, TEST_POS, TEST_NEG, tmp_path)
    assert r.verdict == "lexical_only", r.to_markdown()
    assert r.findings["test_accuracy"] >= 0.9


def test_a_probe_on_labels_unrelated_to_the_test_never_passes(gpt2, tmp_path):
    """Each half mixes sentiments, so nothing it learns can carry over to the sentiment test.

    The halves still differ in their first words ("This/The/One" vs "A/I"), so the probe
    fits them perfectly under cross-validation, which is the overfitting this pathway is
    for. It has to be caught at transfer, not reported as a monitor."""
    mixed = SENT_POS[::2] + SENT_NEG[::2], SENT_POS[1::2] + SENT_NEG[1::2]
    r = probe_monitor.run(gpt2, *mixed, TEST_POS, TEST_NEG, tmp_path)
    assert r.verdict in {"no_signal", "does_not_transfer"}, r.to_markdown()


REVIEWS = ["the food was great and the staff were kind", "I loved the movie", "the hotel was lovely",
           "the concert was amazing", "the book was wonderful", "the trip was fantastic",
           "the class was excellent", "the game was fun"]
FEMALE_NAMES = [a for a, _ in NAMES]
MALE_NAMES = [b for _, b in NAMES]


def test_sentiment_does_not_rely_on_the_reviewers_name(gpt2, tmp_path):
    r = spurious_audit.run(
        gpt2, [f"Review by {a}: {x}. Overall the review is" for a, x in zip(FEMALE_NAMES, REVIEWS)],
        [f"Review by {b}: {x}. Overall the review is" for b, x in zip(MALE_NAMES, REVIEWS)],
        [" positive"] * 8, [" negative"] * 8, tmp_path)
    assert r.verdict == "no_reliance", r.to_markdown()


def test_gendered_names_pull_on_a_stated_occupation_and_erasure_removes_it(gpt2, tmp_path):
    r = spurious_audit.run(
        gpt2, [f"My friend {b} has always worked as a pilot. Everyone knows that {b} is a" for b in MALE_NAMES],
        [f"My friend {a} has always worked as a pilot. Everyone knows that {a} is a" for a in FEMALE_NAMES],
        [" pilot"] * 8, [" secretary"] * 8, tmp_path, control_texts=CONTROL_TEXT)
    assert r.verdict in {"removable", "entangled"}, r.to_markdown()
    assert r.findings["fraction_removed"] - r.findings["random_fraction_removed"] >= 0.5


FACTS = [("The Eiffel Tower is located in the city of", "The Eiffel Tower", " Paris"),
         ("The Colosseum is located in the city of", "The Colosseum", " Rome"),
         ("The Statue of Liberty is located in the city of", "The Statue of Liberty", " New"),
         ("Michael Jordan played the sport of", "Michael Jordan", " basketball"),
         ("The capital of Japan is the city of", "Japan", " Tokyo"),
         ("Big Ben is located in the city of", "Big Ben", " London")]


def test_the_fact_moves_from_the_subject_to_the_final_token(gpt2, tmp_path):
    r = factual_recall.run(gpt2, *[[f[i] for f in FACTS] for i in range(3)], tmp_path)
    assert r.verdict in {"subject_mlp_recall", "late_site", "not_localized"}, r.to_markdown()
    resid = torch.tensor(r.steps["causal_tracing"].data["effects"]["resid"])
    groups = r.steps["causal_tracing"].data["groups"]
    subj, final = groups.index("subject_last"), groups.index("final")
    half = len(resid) // 2
    # Early: the fact is carried at the subject's last token, not the final one.
    assert resid[:half, subj].mean() > resid[:half, final].mean() + 0.1
    # Late: it has moved to the final token.
    assert resid[-1, final] > 0.9 and resid[-1, final] > resid[-1, subj]
    # And it peaks at the subject before the final token overtakes it.
    overtaken = next(l for l in range(len(resid)) if resid[l, final] > resid[l, subj])
    assert int(resid[:, subj].argmax()) < overtaken


EDITS = [("Michael Jordan played the sport of", "Michael Jordan", " basketball", " soccer",
          "The athlete Michael Jordan was famous for playing", "Tiger Woods played the sport of", " golf"),
         ("The capital of Japan is the city of", "Japan", " Tokyo", " Berlin",
          "Japan has its capital in the city of", "The capital of France is the city of", " Paris"),
         ("The Eiffel Tower is located in the city of", "The Eiffel Tower", " Paris", " Rome",
          "To see the Eiffel Tower, travel to the city of", "Big Ben is located in the city of", " London")]


def test_rome_edits_work_mid_model_and_not_at_the_last_layer(gpt2, tmp_path):
    r = editing_comparison.run(gpt2, *[[e[i] for e in EDITS] for i in range(7)], tmp_path,
                               layers=[2, 5, 8, 11])
    scores = r.findings["edit_scores"]
    assert scores["5"]["efficacy"] >= 2 / 3, r.to_markdown()
    # The last layer's MLP at the subject has nothing after it to carry the edit to the answer.
    assert scores["11"]["efficacy"] == 0.0


def test_a_change_planted_in_one_block_is_located(gpt2, tmp_path):
    import copy
    other = copy.deepcopy(gpt2)
    with torch.no_grad():
        w = other.blocks[5].mlp.W_out
        w.add_(0.5 * w.std() * torch.randn_like(w))
    r = model_diffing.run(gpt2, other, [f[0] for f in FACTS] + IOI[:4], tmp_path)
    assert r.verdict == "localized", r.to_markdown()
    assert max(r.findings["kl_closed_by_component"].items(), key=lambda kv: kv[1])[0] == "block5"
