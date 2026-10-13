"""Mechanics of the seesaw.interp techniques, on tiny random models.

Every test here holds for any weights, so it runs in seconds and needs no
download. Where a technique has an exact answer on a model simple enough,
it's checked against it: on a one-layer attention-only model with no
LayerNorm the logit diff is linear in each head's output, so attribution
patching must equal real patching exactly.

Skipped when torch / transformer-lens aren't installed (the CI checks tier).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformer_lens")

from transformer_lens import HookedTransformer, HookedTransformerConfig  # noqa: E402

from seesaw.interp._helpers import parse_head, top_heads  # noqa: E402
from seesaw.interp.black_box import counterfactual  # noqa: E402
from seesaw.interp.causal_interventions import (  # noqa: E402
    ablation, attribution_patching, head_patching,
)
from seesaw.interp.feature_extraction import feature_attribution, neuronpedia  # noqa: E402
from seesaw.interp.metrics import logit_diff, recovery  # noqa: E402
from seesaw.interp.model_editing import directional_ablation  # noqa: E402
from seesaw.interp.probing import linear_probe  # noqa: E402
from seesaw.interp.steering import feature_clamping  # noqa: E402
from seesaw.pathways import attribute_audit, circuit_discovery, feature_explain  # noqa: E402
from seesaw.pathways.result import PathwayResult  # noqa: E402

PROMPTS   = ["When Mary and John went to the store, John gave a drink to",
             "When Alice and Bob went to the park, Bob handed a book to"]
CORRUPTED = ["When Mary and John went to the store, Mary gave a drink to",
             "When Alice and Bob went to the park, Alice handed a book to"]
POS, NEG  = [" Mary", " Alice"], [" John", " Bob"]


def _model(**overrides) -> HookedTransformer:
    cfg = dict(n_layers=2, d_model=32, d_head=8, n_heads=4, d_mlp=64, n_ctx=64,
               act_fn="gelu", tokenizer_name="gpt2", normalization_type="LN", seed=0)
    cfg.update(overrides)
    return HookedTransformer(HookedTransformerConfig(**cfg)).eval()


@pytest.fixture(scope="module")
def model():
    return _model()


@pytest.fixture(scope="module")
def linear_model():
    """One attention layer, no MLP, no LayerNorm: logits are linear in each head's z."""
    return _model(n_layers=1, attn_only=True, normalization_type=None)


class FakeSAE:
    """Identity 'SAE' on resid_pre of layer 1: features are the residual coordinates."""

    class _Cfg:
        class metadata:  # noqa: N801
            hook_name = "blocks.1.hook_resid_pre"
            neuronpedia_id = None

    cfg = _Cfg()

    def __init__(self, d_model: int):
        self.W_dec = torch.eye(d_model)

    def encode(self, x):
        return x


# ── helpers and metrics ──────────────────────────────────────────────────────
def test_head_names_round_trip():
    assert parse_head("L9H9") == (9, 9)
    assert top_heads(torch.tensor([[3.0, -5.0], [1.0, 2.0]]), 2) == [("L0H0", 3.0), ("L1H1", 2.0)]
    assert top_heads(torch.tensor([[-1.0, -2.0]]), 3) == []


def test_logit_diff_reads_per_prompt_positions():
    logits = torch.zeros(2, 3, 5)
    logits[0, 1, 2], logits[1, 2, 4] = 4.0, 7.0
    out = logit_diff(logits, [2, 4], [0, 0], torch.tensor([1, 2]))
    assert out.tolist() == [4.0, 7.0]
    assert recovery(patched=1.0, corrupt=0.0, clean=2.0) == 0.5


# ── causal interventions ─────────────────────────────────────────────────────
def test_attribution_patching_is_exact_on_a_linear_model(linear_model, tmp_path):
    atp = attribution_patching.compute(linear_model, PROMPTS, CORRUPTED, POS, NEG, tmp_path)
    heads = [f"L0H{h}" for h in range(linear_model.cfg.n_heads)]
    real = head_patching.compute(linear_model, PROMPTS, CORRUPTED, POS, NEG, heads, tmp_path)
    for h, name in enumerate(heads):
        assert atp.data["head_effects"][0][h] == pytest.approx(real.data["per_head"][name], abs=1e-3)


def test_joint_patching_adds_up_on_a_linear_model(linear_model, tmp_path):
    # Heads' effects only interact through nonlinearities; with none, patching
    # them together must recover exactly the sum of patching each alone.
    heads = [f"L0H{h}" for h in range(linear_model.cfg.n_heads)]
    r = head_patching.compute(linear_model, PROMPTS, CORRUPTED, POS, NEG, heads, tmp_path)
    assert r.data["joint_recovery"] == pytest.approx(sum(r.data["per_head"].values()), abs=1e-3)


def test_ablation_on_a_shortlist_matches_the_full_sweep(model, tmp_path):
    full = ablation.compute(model, PROMPTS, POS, NEG, tmp_path)
    some = ablation.compute(model, PROMPTS, POS, NEG, tmp_path, heads=["L1H2", "L0H1"])
    for layer in range(2):
        for head in range(4):
            expected = full.data["effects"][layer][head] if (layer, head) in {(1, 2), (0, 1)} else 0.0
            assert some.data["effects"][layer][head] == pytest.approx(expected, abs=1e-4)


def test_resample_ablation_from_the_same_prompts_changes_nothing(model, tmp_path):
    r = ablation.compute(model, PROMPTS, POS, NEG, tmp_path,
                         ablation_type="resample", resample_prompts=PROMPTS)
    assert torch.tensor(r.data["effects"]).abs().max().item() < 1e-4


def test_resample_needs_aligned_prompts(model, tmp_path):
    with pytest.raises(ValueError):
        ablation.compute(model, PROMPTS, POS, NEG, tmp_path, ablation_type="resample")


# ── black box, probing, steering, editing ────────────────────────────────────
def test_counterfactual_of_itself_has_no_effect(model, tmp_path):
    r = counterfactual.compute(model, PROMPTS, PROMPTS, POS, NEG, tmp_path)
    assert r.data["mean_effect"] == pytest.approx(0.0, abs=1e-5)


def test_probe_fits_separable_data():
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(40, 16, generator=gen)
    y = (x[:, 0] > 0).float()
    pred = linear_probe._fit_predict(x, y, x, weight_decay=1e-3)
    assert (pred.float() == y).float().mean().item() >= 0.95


def test_directional_ablation_removes_the_component(model):
    unit = torch.nn.functional.normalize(torch.randn(model.cfg.d_model), dim=0)
    (name, project_out), *_ = directional_ablation._hooks(model, unit)
    x = torch.randn(2, 5, model.cfg.d_model)
    assert (project_out(x, None) @ unit).abs().max().item() < 1e-5


def test_removing_a_zero_effect_direction_reports_nothing_removed(model, tmp_path):
    r = directional_ablation.compute(
        model, PROMPTS, POS, NEG, torch.randn(model.cfg.d_model).tolist(), tmp_path,
        counterfactual_prompts=PROMPTS, n_random=1)
    assert r.data["fraction_removed"] == 0.0     # gap is 0, so nothing to remove


# ── features ─────────────────────────────────────────────────────────────────
def test_feature_attribution_sums_to_gradient_times_input(model, tmp_path):
    """With an identity SAE, Σ_f act_f · grad_f is grad · resid over real tokens."""
    sae = FakeSAE(model.cfg.d_model)
    r = feature_attribution.compute(model, sae, PROMPTS, POS, NEG, tmp_path, explain=False)
    saved = {}
    tokens = model.to_tokens(PROMPTS)

    def root(act, hook):
        return act.detach().requires_grad_(True)

    def keep(act, hook):
        saved["r"] = act
        return act

    from seesaw.interp.metrics import last_positions
    pos = last_positions(model, PROMPTS)
    with torch.enable_grad(), model.hooks([("hook_embed", root), (sae.cfg.metadata.hook_name, keep)]):
        m = logit_diff(model(tokens), [model.to_single_token(t) for t in POS],
                       [model.to_single_token(t) for t in NEG], pos).mean()
        (g,) = torch.autograd.grad(m, saved["r"])
    mask = (torch.arange(tokens.shape[1])[None] <= pos[:, None]).float()[..., None]
    expected = (saved["r"].detach() * g * mask).sum().item()
    assert r.data["total_attribution"] == pytest.approx(expected, abs=1e-3)


def test_feature_clamping_to_current_value_is_a_no_op(model, tmp_path):
    sae = FakeSAE(model.cfg.d_model)

    class Shifted(FakeSAE):
        def encode(self, x):           # every feature reads 1.0 ...
            return torch.ones_like(x)

    # ... so clamping to 1.0 changes nothing.
    r = feature_clamping.compute(model, Shifted(model.cfg.d_model), PROMPTS, POS, NEG,
                                 [0, 3], tmp_path, value=1.0, n_random=0)
    assert r.data["joint_change"] == pytest.approx(0.0, abs=1e-5)
    r = feature_clamping.compute(model, sae, PROMPTS, POS, NEG, [0, 3], tmp_path, n_random=2)
    assert set(r.data["per_feature_change"]) == {"0", "3"}


def test_neuronpedia_client_parses_and_caches(monkeypatch, tmp_path):
    monkeypatch.setenv("SEESAW_NEURONPEDIA_CACHE", str(tmp_path))
    calls = []

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"explanations": [{"description": " names "}], "pos_str": ["a"], ' \
                   b'"neg_str": ["b"], "maxActApprox": 3.0}'

    def fake_urlopen(req, timeout):
        calls.append(req.full_url)
        return Resp()

    monkeypatch.setattr(neuronpedia.urllib.request, "urlopen", fake_urlopen)
    f = neuronpedia.get_feature("gpt2-small/8-res-jb", 658)
    assert f["explanation"] == "names" and f["promotes"] == ["a"]
    assert f["url"] == "https://www.neuronpedia.org/gpt2-small/8-res-jb/658"
    neuronpedia.get_feature("gpt2-small/8-res-jb", 658)
    assert len(calls) == 1      # second call served from the cache


# ── pathways (mechanics only; known answers are in test_pathways_gpt2.py) ─────
def test_pathway_result_reports_early_stops():
    r = PathwayResult(pathway="x").stop("no_effect", "nothing moved")
    assert r.verdict == "no_effect" and r.stopped_early == "nothing moved"
    assert "`no_effect`" in r.to_markdown()


def test_pathways_run_end_to_end(model, tmp_path, monkeypatch):
    # Random weights rarely clear the effect thresholds, so lower them: this
    # checks the plumbing, not the science.
    monkeypatch.setattr(circuit_discovery, "EFFECT_MIN", -1e9)
    monkeypatch.setattr(attribute_audit, "EFFECT_MIN", 0.0)
    monkeypatch.setattr(feature_explain, "EFFECT_MIN", -1e9)

    cd = circuit_discovery.run(model, PROMPTS, POS, NEG, tmp_path / "cd", corrupted_prompts=CORRUPTED)
    assert cd.verdict in {"circuit_found", "partial_circuit", "no_circuit"}
    roles = cd.findings
    assert not set(roles["confirmed_writers"]) & set(roles["writes_but_redundant"])
    assert set(roles["necessary_but_indirect"]) <= set(roles["confirmed_heads"])

    aa = attribute_audit.run(model, PROMPTS, CORRUPTED, POS, NEG, tmp_path / "aa")
    assert aa.verdict in {"uses_attribute", "encoded_not_causal", "inconclusive"}
    assert list(aa.steps)[0] == "counterfactual"

    fe = feature_explain.run(model, FakeSAE(model.cfg.d_model), PROMPTS, POS, NEG, tmp_path / "fe")
    assert fe.verdict in {"features_explain", "partial", "not_explained"} or fe.stopped_early
