"""The tools added for Gemma-2, Llama-3.2 and Qwen2.5, and what those models need from the others.

Tiny random models in the architecture those three share (rotary positions,
RMSNorm, gated MLP, grouped-query attention), so it runs in seconds. Where a
technique has an exact answer it's checked against it. Known answers on the
real models are in tests/test_pathways_models.py.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformer_lens")

from transformer_lens import HookedTransformer, HookedTransformerConfig  # noqa: E402

from lens.mcp_server.src.config.settings import GPU_LARGE, GPU_SMALL, gpu_for  # noqa: E402
from lens.mcp_server.src.tools import circuit_tracing as tracing_tool  # noqa: E402
from seesaw.interp.causal_interventions import kv_head_patching  # noqa: E402
from seesaw.interp.circuit_finding import edge_attribution_patching, path_patching  # noqa: E402
from seesaw.interp.circuit_finding._edges import parse_edge, receivers  # noqa: E402
from seesaw.interp.hidden_state_decoding import patchscope  # noqa: E402
from seesaw.interp.steering import feature_steering, steering_vector  # noqa: E402

PROMPTS   = ["When Mary and John went to the store, John gave a drink to",
             "When Alice and Bob went to the park, Bob handed a book to"]
CORRUPTED = ["When Mary and John went to the store, Mary gave a drink to",
             "When Alice and Bob went to the park, Alice handed a book to"]
POS, NEG  = [" Mary", " Alice"], [" John", " Bob"]


@pytest.fixture(scope="module")
def model():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=32, d_head=8, n_heads=4, n_key_value_heads=2, d_mlp=64, n_ctx=64,
        act_fn="silu", gated_mlp=True, positional_embedding_type="rotary", rotary_dim=8,
        normalization_type="RMS", tokenizer_name="gpt2", seed=0)
    return HookedTransformer(cfg).eval()


# ── routing and loading ──────────────────────────────────────────────────────
def test_models_route_to_a_gpu_by_size():
    for name in ("google/gemma-2-2b", "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B",
                 "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B"):
        assert gpu_for(name) == GPU_SMALL, name
    for name in ("Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct"):
        assert gpu_for(name) == GPU_LARGE, name


# ── grouped-query attention ──────────────────────────────────────────────────
def test_key_and_value_receivers_are_kv_heads(model):
    names = {r.name for r in receivers(model)}
    assert {"L1H3.q", "L1KV1.k", "L1KV1.v"} <= names
    assert "L1H3.k" not in names                    # keys belong to the KV head, not a query head
    _, r = parse_edge("L0H2->L1KV1.v", model)
    assert (r.hook, r.head) == ("blocks.1.hook_v_input", 1)


def test_edges_into_kv_heads_can_be_estimated_and_patched(model, tmp_path):
    eap = edge_attribution_patching.compute(model, PROMPTS, CORRUPTED, POS, NEG, tmp_path, top_k=200)
    edges = [e for e, _ in eap.data["top_edges"]]
    assert any(".k" in e or ".v" in e for e in edges)
    kv = [e for e in edges if e.endswith((".k", ".v"))][:3]
    pp = path_patching.compute(model, PROMPTS, CORRUPTED, POS, NEG, kv, tmp_path)
    assert set(pp.data["per_edge"]) == set(kv)


def test_kv_head_patching_names_each_group(model, tmp_path):
    r = kv_head_patching.compute(model, PROMPTS, CORRUPTED, POS, NEG, tmp_path)
    assert set(r.data["per_kv_head"]) == {"L0KV0", "L0KV1", "L1KV0", "L1KV1"}
    assert r.data["per_kv_head"]["L1KV1"]["query_heads"] == ["L1H2", "L1H3"]
    assert r.data["grouped_query"] and r.data["heads_per_group"] == 2


def test_kv_head_patching_with_no_gap_recovers_nothing(model, tmp_path):
    same = kv_head_patching.compute(model, PROMPTS, PROMPTS, POS, NEG, tmp_path)
    assert all(v["both"] == 0.0 for v in same.data["per_kv_head"].values())


# ── feature steering ─────────────────────────────────────────────────────────
class _IdentitySAE:
    def __init__(self, d_model: int, hook: str):
        self.W_dec = torch.eye(d_model)
        meta = type("Meta", (), {"hook_name": hook, "neuronpedia_id": None})
        self.cfg = type("Cfg", (), {"metadata": meta})

    def encode(self, x):
        return x


def test_feature_steering_is_steering_along_the_decoder_direction(model, tmp_path):
    sae = _IdentitySAE(model.cfg.d_model, "blocks.1.hook_resid_pre")    # = blocks.0.hook_resid_post
    r = feature_steering.compute(model, sae, PROMPTS, POS, NEG, 3, tmp_path, alphas=[0.0, 1.0])
    unit = torch.zeros(model.cfg.d_model)
    unit[3] = r.data["resid_norm"]
    same = steering_vector.compute(model, PROMPTS, POS, NEG, unit.tolist(), 0, tmp_path, alphas=[0.0, 1.0])
    assert r.data["lds"] == pytest.approx(same.data["lds"], abs=1e-3)
    assert r.data["feature"] == 3 and r.data["random_abs_slope"] is not None


def test_feature_steering_refuses_the_embedding_sae(model, tmp_path):
    with pytest.raises(ValueError, match="after the embeddings"):
        feature_steering.compute(model, _IdentitySAE(32, "blocks.0.hook_resid_pre"), PROMPTS, POS, NEG,
                                 0, tmp_path)


# ── patchscope ───────────────────────────────────────────────────────────────
def test_the_logit_lens_at_the_last_layer_is_the_models_own_answer(model, tmp_path):
    r = patchscope.compute(model, PROMPTS, tmp_path, positive_tokens=POS)
    assert len(r.data["per_layer"]) == model.cfg.n_layers
    last = r.data["per_layer"][-1]["logit_lens"]["answer_prob"]
    with torch.no_grad():
        p = [model(model.to_tokens(t))[0, -1].softmax(-1)[model.to_single_token(a)].item()
             for t, a in zip(PROMPTS, POS)]
    assert last == pytest.approx(sum(p) / len(p), abs=1e-3)
    assert r.plot_paths and r.plot_paths[0].exists()


def test_patchscope_without_answers_lists_tokens(model, tmp_path):
    r = patchscope.compute(model, PROMPTS, tmp_path)
    assert len(r.data["per_layer"][0]["patchscope"]["top_tokens"]) == 5
    assert r.data["first_layer_patchscope"] is None


# ── circuit tracing's model ──────────────────────────────────────────────────
@pytest.mark.parametrize("name,expected", [("gemma-2-2b", "gemma-2-2b"), ("Qwen3-4B", "qwen3-4b"),
                                           ("Llama-3.2-1B", "gemma-2-2b")])
def test_circuit_tracing_traces_on_the_plans_model_when_neuronpedia_can(name, expected, model, tmp_path,
                                                                       monkeypatch):
    seen = []

    def fake(output_dir, prompt, slug, model_id, allow_generate, top_k):
        seen.append(model_id)
        from seesaw.interp.result import InterpResult
        return InterpResult(data={})

    monkeypatch.setattr(tracing_tool._circuit_tracing, "compute", fake)
    monkeypatch.setattr(model.cfg, "model_name", name)
    r = tracing_tool.run_circuit_tracing(model, PROMPTS, output_dir=tmp_path)
    assert seen == [expected]
    assert ("note" in r.data) == (expected != tracing_tool.GRAPH_MODELS.get(name))
