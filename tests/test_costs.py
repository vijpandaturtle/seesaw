"""Job costs: what's been spent, priced from the timeline, and what approving the next stage will cost.

Prices are checked against hand arithmetic; estimates against past stages'
cost records and the documented fallbacks. No database, no network.
"""

from __future__ import annotations

import pytest

from shared import costs


def llm(stage, model, inp, out, cache_read=0, cache_write=0):
    return {"stage": stage, "kind": "llm", "name": model, "duration_ms": 1000,
            "data": {"model": model, "usage": {
                "input_tokens": inp, "output_tokens": out,
                "input_token_details": {"cache_read": cache_read, "cache_creation": cache_write}}}}


def stage(name, seconds):
    return {"stage": name, "kind": "stage", "name": name, "duration_ms": seconds * 1000, "data": None}


def experiment(seconds):
    return {"stage": "lens", "kind": "experiment", "name": "x", "duration_ms": seconds * 1000, "data": {}}


@pytest.fixture(autouse=True)
def _local(monkeypatch):
    monkeypatch.delenv("SEESAW_LENS_BACKEND", raising=False)


def test_a_call_is_priced_at_its_models_list_price():
    # 1M in + 1M out on Opus 4.5: $5 + $25.
    assert costs.llm_call_usd("claude-opus-4-5", {"input_tokens": 10**6, "output_tokens": 10**6}) == 30.0
    # Dated ids price as the model.
    assert costs.llm_call_usd("claude-sonnet-4-5-20250929", {"input_tokens": 10**6}) == 3.0


def test_cached_tokens_are_priced_as_reads_and_writes_not_fresh_input():
    usage = {"input_tokens": 10**6, "output_tokens": 0,
             "input_token_details": {"cache_read": 600_000, "cache_creation": 400_000}}
    # Sonnet 4.5: 0.6M reads at $0.30, 0.4M 5-minute writes at $3.75, no fresh input.
    assert costs.llm_call_usd("claude-sonnet-4-5", usage) == pytest.approx(0.18 + 1.5)


def test_an_unknown_model_is_reported_not_priced_at_zero():
    assert costs.llm_call_usd("gpt-9", {"input_tokens": 100}) is None
    spent = costs.spent([stage("scout", 10), llm("scout", "gpt-9", 100, 10)], ("scout",))
    assert spent["unpriced_models"] == ["gpt-9"]


def test_experiments_cost_gpu_time_only_when_they_run_on_modal(monkeypatch):
    events = [stage("lens", 100), llm("lens", "claude-sonnet-4-5", 10_000, 1_000), experiment(4), experiment(6)]
    local = costs.stage_cost(events, "lens")
    assert local["gpu_seconds"] == 0 and local["gpu_usd"] == 0
    monkeypatch.setenv("SEESAW_LENS_BACKEND", "modal")
    remote = costs.stage_cost(events, "lens")
    # 10 s of experiments, plus loading and the warm tail Modal bills.
    assert remote["gpu_seconds"] == 10 + costs.LENS_COLD_START_SECONDS + costs.LENS_IDLE_SECONDS
    assert remote["gpu_usd"] == pytest.approx(remote["gpu_seconds"] * costs.GPU_PER_SECOND["A10G"], abs=1e-4)
    assert remote["experiments"] == 2
    assert remote["llm_usd"] == pytest.approx(0.03 + 0.015)


def test_spent_covers_only_stages_that_ran():
    events = [stage("scout", 60), llm("scout", "claude-opus-4-5", 100_000, 5_000)]
    spent = costs.spent(events, ("scout", "lens", "quill"))
    assert [s["stage"] for s in spent["stages"]] == ["scout"]
    assert spent["total_usd"] == pytest.approx(0.625 + 60 * costs._runner_rate(), abs=1e-4)


def test_with_no_history_the_estimate_says_its_a_default():
    est = costs.estimate("quill", [])
    assert est["llm_usd"] == costs.DEFAULTS["quill"]["llm_usd"]
    assert "default" in est["basis"][0]


def test_the_estimate_follows_past_stages():
    history = [{"stage": "quill", "llm_usd": u, "seconds": 90} for u in (0.1, 0.3, 0.2)]
    est = costs.estimate("quill", history + [{"stage": "lens", "llm_usd": 9, "seconds": 9}])
    assert est["llm_usd"] == 0.2 and "last 3 quill stages" in est["basis"][0]


def test_a_lens_estimate_scales_with_the_experiments_the_plan_names(monkeypatch):
    monkeypatch.setenv("SEESAW_LENS_BACKEND", "modal")
    fixed = costs.LENS_COLD_START_SECONDS + costs.LENS_IDLE_SECONDS
    history = [{"stage": "lens", "llm_usd": 0.1, "seconds": 100, "experiments": 4,
                "gpu_seconds": 4 * 5 + fixed}]
    plan = "Run `ablation`, then `head_patching` and `circuit_evaluation`; finally logit_lens."
    est = costs.estimate("lens", history, plan_text=plan)
    n = 4 + costs.LENS_FOLLOWUPS
    assert est["gpu_seconds"] == pytest.approx(n * 5 + fixed)
    assert est["llm_usd"] == pytest.approx(0.1 * n / 4)
    assert any("names 4 Lens tools" in b for b in est["basis"])


def test_a_lens_run_off_modal_costs_no_gpu():
    assert costs.estimate("lens", [], plan_text="`ablation`")["gpu_seconds"] == 0


def test_describe_reads_like_a_receipt(monkeypatch):
    monkeypatch.setenv("SEESAW_LENS_BACKEND", "modal")
    cost = costs.summary([stage("scout", 60), llm("scout", "claude-opus-4-5", 100_000, 5_000)],
                         ("scout", "lens", "quill"), "lens", [], plan_text="`ablation`")
    lines = costs.describe(cost)
    assert lines[0].startswith("spent so far  $0.63")
    assert lines[1].startswith("approving lens  ~$") and "GPU" in lines[1]


def test_amounts_are_written_plainly():
    assert costs.usd(1.234) == "$1.23" and costs.usd(0.001) == "<$0.01" and costs.usd(None) == "?"


def test_each_model_is_priced_on_the_gpu_it_runs_on(monkeypatch):
    monkeypatch.setenv("SEESAW_LENS_BACKEND", "modal")
    fixed = costs.LENS_COLD_START_SECONDS + costs.LENS_IDLE_SECONDS
    small = {**experiment(10), "data": {"model": "Qwen/Qwen2.5-0.5B"}}
    large = {**experiment(10), "data": {"model": "Qwen/Qwen2.5-7B"}}
    record = costs.stage_cost([stage("lens", 60), small, large], "lens")
    assert record["gpu"] == "A100-80GB, A10G"
    # Both containers load and idle; each is billed at its own rate.
    assert record["gpu_usd"] == pytest.approx(
        (10 + fixed) * costs.GPU_PER_SECOND["A10G"] + (10 + fixed) * costs.GPU_PER_SECOND["A100-80GB"],
        abs=1e-4)
    assert record["gpu_fixed_seconds"] == 2 * fixed


def test_an_estimate_runs_the_plan_on_every_model(monkeypatch):
    monkeypatch.setenv("SEESAW_LENS_BACKEND", "modal")
    one = costs.estimate("lens", [], plan_text="`ablation`", models=["Qwen/Qwen2.5-0.5B"])
    two = costs.estimate("lens", [], plan_text="`ablation`",
                         models=["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-7B"])
    assert two["gpu"] == "A100-80GB, A10G"
    assert two["llm_usd"] == pytest.approx(2 * one["llm_usd"])
    assert two["gpu_usd"] > 2 * one["gpu_usd"]       # the second model is on the dearer GPU
