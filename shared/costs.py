"""What a job has cost so far, and what the stage behind its approval gate will cost.

Shown at every human gate (the dashboard's approval box and `seesaw jobs`), so
whoever approves a stage sees the money first. Three things are billed:

- Claude tokens, from each LLM call on the job's timeline (model and token
  usage, recorded by shared/db/events.py), at Anthropic's list prices.
- GPU time on the Lens app (lens/modal_app.py), from each experiment's
  duration on the timeline, plus the time a container spends loading the
  model and staying warm after its last call (Modal bills both).
- The runner (orchestrator/modal_app.py), a CPU container for the whole stage.

Spent amounts are computed from this job's own timeline. Estimates for the next
stage come from the cost records of past stages (one COST event per finished
stage), falling back to defaults measured on early jobs. Both are estimates: the
bills from Anthropic and Modal are the record.

No torch import: the worker, the CLI and the dashboard's API all read this.
"""

from __future__ import annotations

import os
import re
import time

# ── Prices ────────────────────────────────────────────────────────────────────
# USD per million tokens: (input, 5-minute cache write, 1-hour cache write,
# cache read, output). Anthropic first-party list prices, checked 2026-11-04 at
# https://platform.claude.com/docs/en/about-claude/pricing.
CLAUDE_PRICES: dict[str, tuple[float, float, float, float, float]] = {
    "claude-opus-4-5":   (5.00, 6.25, 10.00, 0.50, 25.00),
    "claude-sonnet-4-5": (3.00, 3.75, 6.00, 0.30, 15.00),
    "claude-haiku-4-5":  (1.00, 1.25, 2.00, 0.10, 5.00),
    "claude-opus-5-5":   (4.00, 5.00, 8.00, 0.20, 20.00),
    "claude-sonnet-5-5": (2.00, 2.50, 4.00, 0.10, 10.00),
    "claude-haiku-5-5":  (0.10, 0.125, 0.20, 0.01, 0.50),   # prompts up to 100K tokens
}

# USD per second on Modal, checked 2026-11-04 at https://modal.com/pricing.
# Modal lists the A10G as "A10".
GPU_PER_SECOND = {"A10G": 0.000306, "A100-80GB": 0.000694, "A100-40GB": 0.000583,
                  "L4": 0.000222, "H100": 0.001097}
CPU_CORE_PER_SECOND = 0.0000131
MEMORY_GIB_PER_SECOND = 0.00000222

# The Lens app's GPU (lens/modal_app.py GPU) and how long a container stays
# warm after its last call (scaledown_window): billed even if nothing else runs.
LENS_GPU = "A10G"
LENS_IDLE_SECONDS = 600
# Loading the container and the model's weights before the first experiment.
LENS_COLD_START_SECONDS = 60
# What the runner container is costed at while a stage runs. Modal bills the
# larger of what's reserved and what's used; a stage mostly waits on the network.
RUNNER_CORES, RUNNER_MEMORY_GIB = 1.0, 1.0

# Per-stage defaults for when no past stage has a cost record, measured on the
# first jobs (gpt2): Scout ~$0.80 of Opus 4.5, Lens ~$0.10 of Sonnet 4.5 and
# ~8 experiments of ~4 s, Quill ~$0.17 of Opus 4.5; stages take 1.5-3 minutes.
DEFAULTS = {
    "scout": {"llm_usd": 0.80, "seconds": 120},
    "lens":  {"llm_usd": 0.10, "seconds": 160, "experiments": 8, "experiment_seconds": 4.0},
    "quill": {"llm_usd": 0.17, "seconds": 95},
}
# Lens runs up to this many follow-ups on top of the plan (lens config MAX_FOLLOWUPS).
LENS_FOLLOWUPS = 2
# Past stages read for an estimate.
HISTORY = 20


def llm_call_usd(model: str, usage: dict) -> float | None:
    """The list price of one LLM call, or None when the model has no price here."""
    price = CLAUDE_PRICES.get(_model_key(model))
    if price is None:
        return None
    details = usage.get("input_token_details") or {}
    cache_read = details.get("cache_read") or 0
    write_1h = details.get("ephemeral_1h_input_tokens") or 0
    write_5m = (details.get("cache_creation") or 0) - write_1h
    # LangChain counts cached and cache-written tokens inside input_tokens.
    fresh = max((usage.get("input_tokens") or 0) - cache_read - write_5m - write_1h, 0)
    inp, w5, w1, read, out = price
    return (fresh * inp + write_5m * w5 + write_1h * w1 + cache_read * read
            + (usage.get("output_tokens") or 0) * out) / 1e6


def _model_key(model: str) -> str:
    """'claude-opus-4-5-20251101' and 'anthropic/claude-opus-4-5' both price as claude-opus-4-5."""
    name = (model or "").split("/")[-1]
    return re.sub(r"-\d{8}$", "", name)


def gpu_for(model: str | None) -> str:
    """The GPU Lens runs `model` on (shared/gpus.py), or LENS_GPU when there's no model to go by."""
    if not model:
        return LENS_GPU
    from shared.gpus import gpu_for as route

    return route(model)


def _gpu_rate(gpu: str) -> float:
    return GPU_PER_SECOND.get(gpu, GPU_PER_SECOND[LENS_GPU])


def _gpu_usd(by_gpu: dict[str, float]) -> float:
    return sum(seconds * _gpu_rate(gpu) for gpu, seconds in by_gpu.items())


def _runner_rate() -> float:
    return RUNNER_CORES * CPU_CORE_PER_SECOND + RUNNER_MEMORY_GIB * MEMORY_GIB_PER_SECOND


def _on_gpu() -> bool:
    """Whether Lens's experiments go to the GPU app (and cost GPU time) rather than run in-process."""
    return (os.environ.get("SEESAW_LENS_BACKEND") or "local").strip().lower() == "modal"


def _pathway_on_gpu() -> bool:
    """Pathway steps run on the GPU app once they go through Lens's registry (seesaw.pathways.tools)."""
    try:
        import importlib.util

        return _on_gpu() and importlib.util.find_spec("seesaw.pathways.tools") is not None
    except (ImportError, ValueError):
        return False


# ── Spent ─────────────────────────────────────────────────────────────────────
def stage_cost(events: list[dict], stage: str) -> dict:
    """What one stage cost, from its events (shared.db.events.list_events, with data)."""
    llm_usd, unpriced, tokens_in, tokens_out = 0.0, set(), 0, 0
    experiments, seconds = 0, 0.0
    pathway, pathway_seconds = None, 0.0
    # GPU seconds per GPU type: a job's models can need different GPUs.
    by_gpu: dict[str, float] = {}
    # A step's model is its pathway run's (one run per model).
    pathway_model = {e["id"]: (e.get("data") or {}).get("model") for e in events
                     if e["kind"] == "pathway" and "id" in e}
    for e in events:
        if e.get("stage") != stage:
            continue
        data = e.get("data") or {}
        if e["kind"] == "llm" and data.get("usage"):
            usage = data["usage"]
            tokens_in += usage.get("input_tokens") or 0
            tokens_out += usage.get("output_tokens") or 0
            usd = llm_call_usd(data.get("model") or e["name"], usage)
            if usd is None:
                unpriced.add(data.get("model") or e["name"])
            else:
                llm_usd += usd
        elif e["kind"] == "experiment" and _on_gpu():
            experiments += 1
            gpu = gpu_for(data.get("model"))
            by_gpu[gpu] = by_gpu.get(gpu, 0.0) + (e.get("duration_ms") or 0) / 1000
        elif e["kind"] == "step" and _pathway_on_gpu():
            gpu = gpu_for(pathway_model.get(e.get("parent_id")))
            by_gpu[gpu] = by_gpu.get(gpu, 0.0) + (e.get("duration_ms") or 0) / 1000
        elif e["kind"] == "stage":
            seconds += (e.get("duration_ms") or 0) / 1000
        if e["kind"] == "pathway":
            pathway, pathway_seconds = e["name"], pathway_seconds + (e.get("duration_ms") or 0) / 1000
    # Each GPU's container was loaded for these calls and stayed warm after them.
    fixed = len(by_gpu) * (LENS_COLD_START_SECONDS + LENS_IDLE_SECONDS)
    for gpu in by_gpu:
        by_gpu[gpu] += LENS_COLD_START_SECONDS + LENS_IDLE_SECONDS
    gpu_seconds = sum(by_gpu.values())
    gpu_usd = _gpu_usd(by_gpu)
    runner_usd = seconds * _runner_rate()
    return {
        "stage": stage,
        "llm_usd": round(llm_usd, 4), "tokens_in": tokens_in, "tokens_out": tokens_out,
        "unpriced_models": sorted(unpriced),
        "experiments": experiments, "pathway": pathway, "pathway_seconds": round(pathway_seconds, 1),
        "gpu": ", ".join(sorted(by_gpu)) or None,
        "gpu_seconds": round(gpu_seconds, 1), "gpu_fixed_seconds": fixed, "gpu_usd": round(gpu_usd, 4),
        "seconds": round(seconds, 1), "runner_usd": round(runner_usd, 4),
        "total_usd": round(llm_usd + gpu_usd + runner_usd, 4),
    }


def spent(events: list[dict], stages: tuple[str, ...]) -> dict:
    """Every finished stage's cost, and their total."""
    per_stage = [stage_cost(events, s) for s in stages
                 if any(e.get("stage") == s and e["kind"] == "stage" for e in events)]
    return {"stages": per_stage, "total_usd": round(sum(s["total_usd"] for s in per_stage), 4),
            "unpriced_models": sorted({m for s in per_stage for m in s["unpriced_models"]})}


# ── Estimate ──────────────────────────────────────────────────────────────────
def _median(values: list[float]) -> float:
    values = sorted(values)
    n = len(values)
    return (values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2) if n else 0.0


def planned_experiments(plan_text: str) -> int | None:
    """How many experiments a plan asks for, read as the distinct Lens tools it names; None if none."""
    try:
        from shared.lens_tools import names

        tools = names()
    except Exception:                                # noqa: BLE001 — an estimate, not a check
        return None
    found = {t for t in tools if re.search(rf"\b{re.escape(t)}\b", plan_text or "")}
    return len(found) or None


def estimate(stage: str, history: list[dict], *, plan_text: str | None = None,
             pathway: str | None = None, models: list[str] | None = None) -> dict:
    """What `stage` will likely cost.

    Args:
        stage: The stage behind the gate.
        history: Cost records of past stages (stage_cost dicts), any stage; the
            ones for `stage` are used.
        plan_text: Scout's plan, for the lens stage: how many experiments it names.
        pathway: The pathway the lens stage runs first, if any.
        models: The models the job runs on. Each one runs the plan (and the
            pathway) on its own GPU (shared/gpus.py).

    Returns:
        {stage, llm_usd, gpu_seconds, gpu_usd, runner_usd, total_usd, basis}:
        `basis` says what the numbers came from, for the person approving.
    """
    past = [h for h in history if h.get("stage") == stage][:HISTORY]
    default = DEFAULTS.get(stage, {"llm_usd": 0.0, "seconds": 60})
    basis = []
    if past:
        llm_usd = _median([h["llm_usd"] for h in past])
        seconds = _median([h["seconds"] for h in past])
        basis.append(f"median of the last {len(past)} {stage} stage{'s' * (len(past) > 1)}")
    else:
        llm_usd, seconds = default["llm_usd"], default["seconds"]
        basis.append(f"no past {stage} stage has a cost record yet; default figures from early jobs")

    by_gpu: dict[str, float] = {}
    if stage == "lens" and _on_gpu():
        models = models or [None]
        with_exps = [h for h in past if h.get("experiments")]
        per_exp = (_median([(h["gpu_seconds"] - h.get("gpu_fixed_seconds",
                                                      LENS_COLD_START_SECONDS + LENS_IDLE_SECONDS))
                            / h["experiments"] for h in with_exps])
                   if with_exps else default["experiment_seconds"])
        per_exp = max(per_exp, 0.5)
        planned = planned_experiments(plan_text or "")
        n = (planned + LENS_FOLLOWUPS) if planned else default["experiments"]
        if planned:
            basis.append(f"the plan names {planned} Lens tool{'s' * (planned > 1)}, "
                         f"plus up to {LENS_FOLLOWUPS} follow-ups")
            # Lens's LLM work grows with the experiments it interprets.
            past_n = _median([h["experiments"] for h in with_exps]) if with_exps else default["experiments"]
            llm_usd *= n / max(past_n, 1)
        if len(models) > 1:
            basis.append(f"each of the {len(models)} models runs them")
            llm_usd *= len(models)
        pw = 0.0
        if pathway and _pathway_on_gpu():
            past_pw = [h["pathway_seconds"] for h in past if h.get("pathway") == pathway
                       and h.get("pathway_seconds")]
            pw = _median(past_pw) if past_pw else 120.0
            basis.append(f"pathway {pathway}: ~{pw:.0f} s per model"
                         + (" (past runs)" if past_pw else " (default)"))
        for model in models:
            gpu = gpu_for(model)
            by_gpu[gpu] = by_gpu.get(gpu, 0.0) + n * per_exp + pw
        for gpu in by_gpu:
            by_gpu[gpu] += LENS_COLD_START_SECONDS + LENS_IDLE_SECONDS
        basis.append(f"GPU includes ~{LENS_COLD_START_SECONDS} s to load and "
                     f"{LENS_IDLE_SECONDS // 60} min warm after the last experiment, per GPU type")
    gpu_seconds = sum(by_gpu.values())
    gpu_usd = _gpu_usd(by_gpu)
    runner_usd = seconds * _runner_rate()
    return {
        "stage": stage,
        "llm_usd": round(llm_usd, 4),
        "gpu": ", ".join(sorted(by_gpu)) or None,
        "gpu_seconds": round(gpu_seconds, 1), "gpu_usd": round(gpu_usd, 4),
        "runner_usd": round(runner_usd, 4),
        "total_usd": round(llm_usd + gpu_usd + runner_usd, 4),
        "basis": basis,
    }


def summary(events: list[dict], stages: tuple[str, ...], next_stage: str | None,
            history: list[dict], **estimate_kwargs) -> dict:
    """The record stored on the job at a gate: spent so far, and the next stage's estimate."""
    return {
        "computed_at": time.time(),
        "spent": spent(events, stages),
        "next": estimate(next_stage, history, **estimate_kwargs) if next_stage else None,
    }


def usd(value: float | None) -> str:
    """'$0.42', '<$0.01', or '?' for an amount that couldn't be priced."""
    if value is None:
        return "?"
    return "<$0.01" if 0 < value < 0.005 else f"${value:,.2f}"


def describe(cost: dict) -> list[str]:
    """Plain lines for a terminal: what's been spent, and what approving will cost."""
    lines = []
    sp = cost.get("spent") or {}
    if sp.get("stages"):
        parts = ", ".join(f"{s['stage']} {usd(s['total_usd'])}" for s in sp["stages"])
        lines.append(f"spent so far  {usd(sp['total_usd'])}  ({parts})")
    nx = cost.get("next")
    if nx:
        bits = [f"Claude {usd(nx['llm_usd'])}"]
        if nx.get("gpu_seconds"):
            bits.append(f"GPU {usd(nx['gpu_usd'])} (~{nx['gpu_seconds'] / 60:.0f} min {nx['gpu']})")
        bits.append(f"runner {usd(nx['runner_usd'])}")
        lines.append(f"approving {nx['stage']}  ~{usd(nx['total_usd'])}  ({', '.join(bits)})")
        lines += [f"  · {b}" for b in nx.get("basis") or []]
    if sp.get("unpriced_models"):
        lines.append("  · not priced (no price on file): " + ", ".join(sp["unpriced_models"]))
    return lines
