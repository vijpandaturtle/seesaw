"""Pathways run every step through Lens's tool registry, and the tools they added behave.

The runner (seesaw.pathways.tools) sends a step in-process on a loaded model,
in-process on Lens's copy of a named model, or to Modal as an experiment
spec. The Modal path is simulated here by a stand-in that does what
lens/modal_app.py's LensRunner does: JSON in, normalize_tool_kwargs, the
registry tool, JSON out. Every pathway is run through it by model name, so a
step whose arguments wouldn't survive the trip (a Python object, a name the
registry doesn't take) fails here rather than on the GPU app.

Tiny random models, so it checks plumbing, not science; the known answers
are in tests/test_pathways_gpt2.py.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformer_lens")

from transformer_lens import HookedTransformer, HookedTransformerConfig  # noqa: E402

from lens.mcp_server.src.app import model_session, remote  # noqa: E402
from lens.mcp_server.src.models.schemas import ExperimentResult  # noqa: E402
from lens.mcp_server.src.tools import TOOL_REGISTRY, features, normalize_tool_kwargs  # noqa: E402
from lens.mcp_server.src.tools.answer_scores import run_answer_scores  # noqa: E402
from lens.mcp_server.src.tools.directional_ablation import run_directional_ablation  # noqa: E402
from lens.mcp_server.src.tools.facts import run_rome_edits  # noqa: E402
from lens.mcp_server.src.tools.features import run_feature_lookup  # noqa: E402
from lens.mcp_server.src.tools.leace_erasure import run_leace_erasure  # noqa: E402
from seesaw.interp.feature_extraction import neuronpedia  # noqa: E402
from seesaw.pathways import (  # noqa: E402
    circuit_discovery, concept_direction, editing_comparison, factual_recall, feature_explain,
    model_diffing, probe_monitor, spurious_audit,
)
from seesaw.pathways.tools import ToolFailed, Tools  # noqa: E402

NAMES = [("Mary", "John"), ("Alice", "Bob"), ("Sarah", "Tom"), ("Anna", "James"),
         ("Kate", "Mark"), ("Lisa", "David")]
PAIRS_A = [f"When {a} and {b} went to the store, {b} gave a drink to" for a, b in NAMES]
PAIRS_B = [f"When {a} and {b} went to the store, {a} gave a drink to" for a, b in NAMES]
POS, NEG = [f" {a}" for a, _ in NAMES], [f" {b}" for _, b in NAMES]
FACTS = ["The Eiffel Tower is located in the city of", "The Colosseum is located in the city of"]
SUBJECTS = ["The Eiffel Tower", "The Colosseum"]
ANSWERS, NEW = [" Paris", " Rome"], [" London", " Berlin"]


def _tiny(seed: int = 0) -> HookedTransformer:
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=32, d_head=8, n_heads=4, d_mlp=64, n_ctx=64,
        act_fn="gelu", tokenizer_name="gpt2", normalization_type="LN", seed=seed)
    return HookedTransformer(cfg).eval()


@pytest.fixture(scope="module")
def model():
    return _tiny()


@pytest.fixture(autouse=True)
def _default_backends(monkeypatch):
    monkeypatch.delenv("SEESAW_INTERP_BACKEND", raising=False)
    monkeypatch.delenv("SEESAW_LENS_BACKEND", raising=False)


# ── the runner ───────────────────────────────────────────────────────────────
def test_in_process_data_is_what_a_remote_run_would_return(model, tmp_path):
    """JSON round-tripped, so int keys become strings exactly as they would over the wire."""
    r = Tools(model, tmp_path)("ablation", PAIRS_A[:2], positive_tokens=POS[:2], negative_tokens=NEG[:2])
    assert r.data == json.loads(json.dumps(r.data))
    assert isinstance(r.data["effects"], list)


def test_a_failing_step_names_its_tool(model, tmp_path):
    with pytest.raises(ToolFailed, match="head_patching"):
        Tools(model, tmp_path)("head_patching", PAIRS_A[:2], corrupted_prompts=PAIRS_B[:2],
                               positive_tokens=POS[:2], negative_tokens=NEG[:2], heads=["L9H9"])


def test_a_named_model_runs_on_lenss_copy_by_default(model, tmp_path, monkeypatch):
    loaded = []
    monkeypatch.setattr(model_session, "get_model", lambda name: loaded.append(name) or model)
    r = Tools("tiny", tmp_path)("answer_scores", PAIRS_A[:2], positive_tokens=POS[:2])
    assert loaded == ["tiny"] and 0 <= r.data["mean_probability"] <= 1


def test_a_named_model_goes_to_modal_as_a_spec(tmp_path, monkeypatch):
    monkeypatch.setenv("SEESAW_LENS_BACKEND", "modal")
    sent = []

    def fake(spec, output_dir):
        sent.append(spec)
        return ExperimentResult(name=spec["name"], tool=spec["tool"], model_name=spec["model_name"],
                                prompts=spec["prompts"], data={"x": 1})

    monkeypatch.setattr(remote, "run_experiment_remote", fake)
    r = Tools("gpt2", tmp_path)("linear_probe", PAIRS_A, counterfactual_prompts=PAIRS_B)
    assert r.data == {"x": 1}
    assert sent == [{"name": "linear_probe", "tool": "linear_probe", "model_name": "gpt2",
                     "prompts": PAIRS_A, "tool_kwargs": {"counterfactual_prompts": PAIRS_B}}]

    def failed(spec, output_dir):
        return ExperimentResult(name="x", tool=spec["tool"], model_name="gpt2", prompts=[],
                                status="failed", error="boom")

    monkeypatch.setattr(remote, "run_experiment_remote", failed)
    with pytest.raises(ToolFailed, match="boom"):
        Tools("gpt2", tmp_path)("linear_probe", PAIRS_A, counterfactual_prompts=PAIRS_B)


def test_a_loaded_model_never_goes_to_modal(model, tmp_path, monkeypatch):
    monkeypatch.setenv("SEESAW_LENS_BACKEND", "modal")
    monkeypatch.setattr(remote, "run_experiment_remote", lambda *a, **k: pytest.fail("sent to Modal"))
    Tools(model, tmp_path)("answer_scores", PAIRS_A[:2], positive_tokens=POS[:2])


# ── the tools the pathways added ─────────────────────────────────────────────
def test_answer_scores_reads_each_prompts_own_last_token(model, tmp_path):
    short, long = "When Mary and John left, John thanked", PAIRS_A[1]
    own = run_answer_scores(model, [short, long], [" Mary", " Alice"], [" John", " Bob"])
    alone = run_answer_scores(model, [short], [" Mary"], [" John"])
    assert own.data["per_prompt"][0]["logit_diff"] == pytest.approx(alone.data["mean_logit_diff"], abs=1e-3)
    padded = run_answer_scores(model, [short, long], [" Mary", " Alice"], [" John", " Bob"], pos=-1)
    assert padded.data["per_prompt"][0]["logit_diff"] != pytest.approx(own.data["per_prompt"][0]["logit_diff"])


def test_baked_directional_ablation_matches_the_hooks(model, tmp_path):
    r = run_directional_ablation(model, PAIRS_A[1::2], PAIRS_B[1::2], POS[1::2], NEG[1::2], layer=1,
                                 fit_prompts=PAIRS_A[::2], fit_counterfactual_prompts=PAIRS_B[::2],
                                 bake=True, output_dir=tmp_path)
    assert r.data["baked_effect_after"] == pytest.approx(r.data["effect_after"], abs=1e-3)


def test_the_direction_is_fitted_on_the_fit_prompts(model, tmp_path):
    test = (PAIRS_A[1::2], PAIRS_B[1::2], POS[1::2], NEG[1::2])
    in_sample = run_directional_ablation(model, *test, layer=1, output_dir=tmp_path)
    held_out = run_directional_ablation(model, *test, layer=1, fit_prompts=PAIRS_A[::2],
                                        fit_counterfactual_prompts=PAIRS_B[::2], output_dir=tmp_path)
    assert in_sample.data["effect_after"] != held_out.data["effect_after"]


def test_leace_erasure_holds_out_every_other_pair_by_default(model, tmp_path):
    default = run_leace_erasure(model, PAIRS_A, PAIRS_B, POS, NEG, output_dir=tmp_path)
    explicit = run_leace_erasure(model, PAIRS_A[1::2], PAIRS_B[1::2], POS[1::2], NEG[1::2],
                                 fit_prompts=PAIRS_A[::2], fit_counterfactual_prompts=PAIRS_B[::2],
                                 output_dir=tmp_path)
    assert default.data == explicit.data and default.data["n_fit"] == 3


def test_rome_edits_score_every_layer_and_restore_the_weights(model, tmp_path):
    before = [b.mlp.W_out.detach().clone() for b in model.blocks]
    r = run_rome_edits(model, FACTS, SUBJECTS, ANSWERS, NEW, paraphrase_prompts=FACTS, output_dir=tmp_path)
    assert set(r.data["per_layer"]) == {"0", "1"} and str(r.data["best_layer"]) in r.data["per_layer"]
    assert all(torch.equal(b.mlp.W_out, w) for b, w in zip(model.blocks, before))


def test_feature_lookup_labels_features_from_neuronpedia(model, monkeypatch):
    monkeypatch.setattr(features, "_sae", lambda m, layer: (8, "rel", "id", "gpt2-small/8-res-jb"))

    def get_feature(source, index):
        if index == 2:
            raise neuronpedia.NeuronpediaError("down")
        return {"index": index, "explanation": f"feature {index}", "url": "u"}

    monkeypatch.setattr(neuronpedia, "get_feature", get_feature)
    r = run_feature_lookup(model, ["x"], [1, 2], layer=8)
    assert [f["explanation"] for f in r.data["features"]] == ["feature 1", None]
    assert "down" in r.data["features"][1]["error"]


class _IdentitySAE:
    """Features are the residual coordinates at blocks.<layer>.hook_resid_pre; not on Neuronpedia."""

    def __init__(self, d_model: int, layer: int):
        self.W_dec = torch.eye(d_model)
        meta = type("Meta", (), {"hook_name": f"blocks.{layer}.hook_resid_pre", "neuronpedia_id": None})
        self.cfg = type("Cfg", (), {"metadata": meta})

    def encode(self, x):
        return x


# ── every pathway, by model name, through a stand-in for Modal ───────────────
@pytest.fixture
def via_modal(model, tmp_path, monkeypatch):
    """Route by-name steps through a fake of LensRunner.run_experiment, running on `model`."""
    monkeypatch.setenv("SEESAW_LENS_BACKEND", "modal")
    other = _tiny(seed=1)
    specs = []

    def run_remote(spec, output_dir):
        spec = json.loads(json.dumps(spec))           # only JSON crosses
        specs.append(spec)
        kwargs, missing, dropped = normalize_tool_kwargs(spec["tool"], spec["tool_kwargs"], len(spec["prompts"]))
        assert not missing and not dropped, (spec["tool"], missing, dropped)
        if "other_model" in kwargs:
            kwargs["other_model"] = other if kwargs["other_model"] == "other" else model
        result = TOOL_REGISTRY[spec["tool"]](model=model, prompts=spec["prompts"],
                                             output_dir=output_dir, **kwargs)
        return ExperimentResult(name=spec["name"], tool=spec["tool"], model_name="tiny",
                                prompts=spec["prompts"], data=json.loads(json.dumps(result.data)),
                                plot_paths=result.plot_paths)

    monkeypatch.setattr(remote, "run_experiment_remote", run_remote)
    monkeypatch.setattr(Tools, "n_layers", property(lambda self: model.cfg.n_layers))
    return specs


def test_every_pathway_runs_by_name_through_modal(via_modal, model, tmp_path, monkeypatch):
    for module, attr, value in [
        (circuit_discovery, "EFFECT_MIN", -1e9), (concept_direction, "EFFECT_MIN", 0.0),
        (feature_explain, "EFFECT_MIN", -1e9), (spurious_audit, "TASK_MIN", -1e9),
        (spurious_audit, "RELIANCE_MIN", 0.0), (factual_recall, "P_MIN", 0.0),
        (factual_recall, "CORRUPTION_MIN", -1e9), (probe_monitor, "SIGNAL_MARGIN", -1e9),
    ]:
        monkeypatch.setattr(module, attr, value)
    monkeypatch.setattr(features, "_load", lambda m, layer: (layer, _IdentitySAE(model.cfg.d_model, layer)))

    name = "gpt2"         # its tokenizer is the tiny model's; every forward pass runs on `model`
    runs = {
        "circuit_discovery": circuit_discovery.run(name, PAIRS_A, POS, NEG, tmp_path / "cd",
                                                   corrupted_prompts=PAIRS_B),
        "concept_direction": concept_direction.run(name, PAIRS_A, PAIRS_B, POS, NEG, tmp_path / "cdir"),
        "probe_monitor": probe_monitor.run(name, PAIRS_A, PAIRS_B, PAIRS_A[:2], PAIRS_B[:2], tmp_path / "pm"),
        "spurious_audit": spurious_audit.run(name, PAIRS_A, PAIRS_B, POS, NEG, tmp_path / "sa"),
        "factual_recall": factual_recall.run(name, FACTS, SUBJECTS, ANSWERS, tmp_path / "fr"),
        "editing_comparison": editing_comparison.run(name, FACTS, SUBJECTS, ANSWERS, NEW, FACTS,
                                                     FACTS[::-1], ANSWERS[::-1], tmp_path / "ec"),
        "model_diffing": model_diffing.run(name, "other", PAIRS_A, tmp_path / "md"),
        "feature_explain": feature_explain.run(name, [1], PAIRS_A, POS, NEG, tmp_path / "fe",
                                               search_neuronpedia=False),
    }
    for pathway, r in runs.items():
        assert r.verdict, (pathway, r.to_markdown())
    assert "bake_check" in runs["concept_direction"].steps
    assert runs["model_diffing"].verdict in {"localized", "distributed"}
    tools_used = {s["tool"] for s in via_modal}
    assert {"answer_scores", "mlp_patching", "leace_erasure", "probe_transfer", "causal_tracing",
            "rome_edits", "model_diff", "feature_attribution", "feature_clamping"} <= tools_used
