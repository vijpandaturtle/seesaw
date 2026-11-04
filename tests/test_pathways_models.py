"""Known-answer tests on the models Lens targets: Gemma-2-2B, Llama-3.2 and Qwen2.5.

Each pathway runs by model name, step by step through Lens's registry, so with
SEESAW_LENS_BACKEND=modal every forward pass runs on the GPU app (point
SEESAW_MODAL_APP at a test copy, e.g. seesaw-lens-dev). What's checked is
what the literature found across model families, not head-level details from
one model:

- circuit_discovery finds IOI heads: there is a behaviour, and heads that
  each carry part of it (Wang et al. 2022). Whether a few heads are the whole
  circuit is model-specific: on Qwen2.5-1.5B the confirmed heads alone keep
  under 10% of it, so the verdict is partial_circuit there.
- concept_direction finds the occupation→pronoun bias (Vig et al. 2020) on
  held-out pairs, and doesn't flag an attribute that changes nothing.
- spurious_audit finds reviewers' names move sentiment far less than the task
  does (GPT-2 small: not at all; Qwen2.5-1.5B: 0.36 logits against a 3.3-logit
  task, consistently in one direction), and finds
  (and erases far beyond random erasers) gendered names' pull on a stated
  occupation.
- factual_recall shows the residual-stream hand-off of Meng et al. 2022: the
  subject's last token carries the fact early, the final token late.
- editing_comparison: ROME edits work mid-model and can't at the last layer.
- probe_monitor: a sentiment probe transfers to product reviews.
- model_diffing: instruction tuning changes the base model's predictions.
- feature_explain (models with published SAEs): SAE features explain IOI.

Opt-in, it needs the models (Gemma and Llama are gated: the GPU app needs an
HF token) and a GPU:

    SEESAW_MODEL_TESTS=1 SEESAW_LENS_BACKEND=modal SEESAW_MODAL_APP=seesaw-lens-dev \\
        pytest tests/test_pathways_models.py

SEESAW_TEST_MODELS picks the models (comma-separated).
"""

from __future__ import annotations

import os

import pytest

if not os.environ.get("SEESAW_MODEL_TESTS"):
    pytest.skip("set SEESAW_MODEL_TESTS=1 to run known-answer tests on the target models",
                allow_module_level=True)

torch = pytest.importorskip("torch")

from seesaw.pathways import (  # noqa: E402
    catalog, circuit_discovery, concept_direction, editing_comparison, factual_recall,
    feature_explain, model_diffing, probe_monitor, spurious_audit,
)
from seesaw.pathways.tools import Tools  # noqa: E402

MODELS = (os.environ.get("SEESAW_TEST_MODELS")
          or "google/gemma-2-2b,meta-llama/Llama-3.2-1B,Qwen/Qwen2.5-1.5B").split(",")
# Each base model's instruction-tuned release: what model_diffing compares it with.
TUNED = {"google/gemma-2-2b": "google/gemma-2-2b-it",
         "meta-llama/Llama-3.2-1B": "meta-llama/Llama-3.2-1B-Instruct",
         "meta-llama/Llama-3.2-3B": "meta-llama/Llama-3.2-3B-Instruct"}

NAMES = [("Mary", "John"), ("Alice", "Bob"), ("Sarah", "Tom"), ("Anna", "James"),
         ("Kate", "Mark"), ("Lisa", "David"), ("Emma", "Paul"), ("Laura", "Peter")]
IOI      = [f"When {a} and {b} went to the store, {b} gave a drink to" for a, b in NAMES]
IOI_CORR = [f"When {a} and {b} went to the store, {a} gave a drink to" for a, b in NAMES]
IOI_ABC  = [f"When {a} and {b} went to the store, Mike gave a drink to" for a, b in NAMES]
IO, S    = [f" {a}" for a, _ in NAMES], [f" {b}" for _, b in NAMES]

FEMALE = ["nurse", "secretary", "nanny", "librarian", "dancer", "maid", "teacher", "cashier"]
MALE   = ["engineer", "mechanic", "carpenter", "plumber", "pilot", "farmer", "soldier", "driver"]
CONTROL_TEXT = ["The weather today is mild and the park was full of people walking their dogs."]

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

REVIEWS = ["the food was great and the staff were kind", "I loved the movie", "the hotel was lovely",
           "the concert was amazing", "the book was wonderful", "the trip was fantastic",
           "the class was excellent", "the game was fun"]
FEMALE_NAMES, MALE_NAMES = [a for a, _ in NAMES], [b for _, b in NAMES]

FACTS = [("The Eiffel Tower is located in the city of", "The Eiffel Tower", " Paris"),
         ("The Colosseum is located in the city of", "The Colosseum", " Rome"),
         ("Michael Jordan played the sport of", "Michael Jordan", " basketball"),
         ("The capital of Japan is the city of", "Japan", " Tokyo"),
         ("Big Ben is located in the city of", "Big Ben", " London"),
         ("The Brandenburg Gate is located in the city of", "The Brandenburg Gate", " Berlin")]
EDITS = [("Michael Jordan played the sport of", "Michael Jordan", " basketball", " soccer",
          "The athlete Michael Jordan was famous for playing", "Tiger Woods played the sport of", " golf"),
         ("The capital of Japan is the city of", "Japan", " Tokyo", " Berlin",
          "Japan has its capital in the city of", "The capital of France is the city of", " Paris"),
         ("The Eiffel Tower is located in the city of", "The Eiffel Tower", " Paris", " Rome",
          "To see the Eiffel Tower, travel to the city of", "Big Ben is located in the city of", " London")]


@pytest.fixture(scope="module", params=MODELS)
def model(request):
    """The model name, after checking every answer token here is one token for it."""
    name = request.param
    tokens = {"positive_tokens": IO + S + [" she", " he", " pilot", " secretary", " positive", " negative"]
              + [f[2] for f in FACTS] + [e[3] for e in EDITS] + [e[6] for e in EDITS]}
    errors = catalog.token_errors(tokens, name)
    assert not errors, errors
    return name


def test_ioi_circuit_is_found(model, tmp_path):
    r = circuit_discovery.run(model, IOI, IO, S, tmp_path, corrupted_prompts=IOI_CORR,
                              reference_prompts=IOI_ABC)
    assert r.verdict in {"circuit_found", "partial_circuit"}, r.to_markdown()
    assert r.findings["confirmed_heads"]


def test_gender_bias_is_found_on_held_out_pairs(model, tmp_path):
    r = concept_direction.run(
        model, [f"The {o} said that" for o in FEMALE], [f"The {o} said that" for o in MALE],
        [" she"] * 8, [" he"] * 8, tmp_path, control_texts=CONTROL_TEXT)
    assert r.verdict in {"uses_direction", "partly_uses_direction"}, r.to_markdown()
    assert r.findings["fraction_removed"] - r.findings["random_fraction_removed"] >= 0.1


def test_irrelevant_attribute_is_not_flagged(model, tmp_path):
    foods_a, foods_b = ["apple", "banana", "orange", "grape"], ["sandwich", "salad", "pizza", "burger"]
    r = concept_direction.run(
        model, [f"The man who ate the {f} said that" for f in foods_a],
        [f"The man who ate the {f} said that" for f in foods_b], [" she"] * 4, [" he"] * 4, tmp_path)
    assert r.verdict == "no_effect", r.to_markdown()


def test_sentiment_does_not_rely_on_the_reviewers_name(model, tmp_path):
    r = spurious_audit.run(
        model, [f"Review by {a}: {x}. Overall the review is" for a, x in zip(FEMALE_NAMES, REVIEWS)],
        [f"Review by {b}: {x}. Overall the review is" for b, x in zip(MALE_NAMES, REVIEWS)],
        [" positive"] * 8, [" negative"] * 8, tmp_path)
    assert r.verdict != "no_task", r.to_markdown()
    assert r.findings["reliance"] < 0.2 * r.findings["task_logit_diff"], r.to_markdown()


def test_gendered_names_pull_on_a_stated_occupation_and_erasure_removes_it(model, tmp_path):
    r = spurious_audit.run(
        model, [f"My friend {b} has always worked as a pilot. Everyone knows that {b} is a" for b in MALE_NAMES],
        [f"My friend {a} has always worked as a pilot. Everyone knows that {a} is a" for a in FEMALE_NAMES],
        [" pilot"] * 8, [" secretary"] * 8, tmp_path, control_texts=CONTROL_TEXT)
    assert r.verdict in {"removable", "entangled", "no_reliance"}, r.to_markdown()
    if r.verdict != "no_reliance":
        assert r.findings["fraction_removed"] - r.findings["random_fraction_removed"] >= 0.5


def test_the_fact_moves_from_the_subject_to_the_final_token(model, tmp_path):
    r = factual_recall.run(model, *[[f[i] for f in FACTS] for i in range(3)], tmp_path)
    assert r.verdict in {"subject_mlp_recall", "late_site", "not_localized"}, r.to_markdown()
    resid = torch.tensor(r.steps["causal_tracing"].data["effects"]["resid"])
    groups = r.steps["causal_tracing"].data["groups"]
    subj, final = groups.index("subject_last"), groups.index("final")
    third = len(resid) // 3
    assert resid[:third, subj].mean() > resid[:third, final].mean() + 0.1
    assert resid[-1, final] > 0.9 and resid[-1, final] > resid[-1, subj]


def test_rome_edits_work_mid_model_and_not_at_the_last_layer(model, tmp_path):
    n = Tools(model, tmp_path).n_layers
    layers = sorted({n // 6, n // 3, n // 2, n - 1})
    r = editing_comparison.run(model, *[[e[i] for e in EDITS] for i in range(7)], tmp_path, layers=layers)
    scores = r.findings["edit_scores"]
    assert max(scores[str(l)]["efficacy"] for l in layers[:-1]) >= 2 / 3, r.to_markdown()
    assert scores[str(n - 1)]["efficacy"] <= 1 / 3


def test_a_sentiment_probe_transfers_to_product_reviews(model, tmp_path):
    r = probe_monitor.run(model, SENT_POS, SENT_NEG, TEST_POS, TEST_NEG, tmp_path)
    assert r.verdict in {"lexical_only", "monitor_works"}, r.to_markdown()
    assert r.findings["test_accuracy"] >= 0.85


def test_instruction_tuning_changes_the_predictions(model, tmp_path):
    tuned = TUNED.get(model, f"{model}-Instruct")
    r = model_diffing.run(model, tuned, [f[0] for f in FACTS] + IOI[:4], tmp_path)
    assert r.verdict != "no_change", r.to_markdown()


def test_sae_features_explain_ioi(model, tmp_path):
    if not catalog.has_saes(model):
        with pytest.raises(ValueError, match="published SAEs"):
            catalog.run_from_params("feature_explain", model,
                                    {"prompts": IOI, "positive_tokens": IO, "negative_tokens": S}, tmp_path)
        return
    r = feature_explain.run(model, catalog.sae_layers(model), IOI, IO, S, tmp_path, search_neuronpedia=False)
    assert r.verdict in {"features_explain", "partial"}, r.to_markdown()
