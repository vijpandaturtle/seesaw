"""Run a job's stages: Scout → Lens → Quill.

A job that names a pathway (params = {"pathway": name}) runs it as part of
the research, and the pathway is binding: Scout only chooses its inputs.
With params = {"pathway": "auto"}, Scout chooses the pathway too, or none;
its choice replaces "auto" in params["pathway"] (or removes it), with
params["pathway_request"] = "auto" kept so a retried Scout stage chooses again
and params["pathway_why"] holding its reason. Either way, the inputs go in
which are checked (and sent back to Scout to fix, at most twice) and saved as
params["pathway_params"] for the human gate to show. Lens's stage then runs
the pathway on exactly those inputs before any of Lens's own experiments,
which are labelled extras, so the same pathway on the same kind of inputs
gives results comparable across models. A rerun (jobs.rerun) starts at Lens
with those inputs already saved, so they're checked again against its model,
and replays the source job's extra experiments exactly as they ran (same
tools, prompts and arguments) instead of having an LLM read the plan again.

    python -m orchestrator.src.worker <job_id>   one job, then exit
    python -m orchestrator.src.worker --poll     claim queued jobs forever

A Lens experiment outlives the request that asked for it, so stages run
here rather than in whatever created the job. The worker takes a job from
the store, runs stages in order, and writes each artifact path back as it
goes.

With auto_approve off (the default), it runs one stage and exits, leaving
the job parked at the next human gate; approving queues the stage again
for a worker to claim. With auto_approve on, one worker runs all three
stages back to back.

--poll is the entry point when nothing starts workers on demand — the
dashboard deployed away from the agents, for instance. Several pollers
can run at once; claims are atomic, so no two take the same job.

Everything printed here (and by the agents) goes to the job's log file
under outputs/job_logs/, mirrored into the row by jobs.sync_log so a
remote dashboard can read it. Alongside it, each stage records a structured
timeline (shared/db/events.py): the agents' nodes, LLM calls and tool calls,
Lens's experiments and the pathway's steps, written as they happen.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
import traceback

from dotenv import load_dotenv

from shared.db import events, jobs
from shared.db.jobs import Job

# Lens honours the plan's stated models, so the user's choice is appended as
# an explicit rule rather than hoping Scout kept to it.
_MODELS_NOTE = (
    "\n\n**Target Models (required, override any other model mentioned above)**: run every "
    "experiment on one of {models}, written exactly so. An experiment the plan puts on any "
    "other model runs on {first}.\n"
)


# Lens runs the plan's experiments after the pathway has already run; this
# tells it not to redo the pathway and what its own experiments are for.
_PATHWAY_DONE = (
    "\n\n**Pathway already run**: the `{name}` pathway (version {version}) has already been run "
    "on the approved inputs; its verdict was {verdict}. Do not re-run it, and do not use its "
    "prompts for an experiment unless the plan lists that experiment on its own. The experiments "
    "you run are extras that add to that result.\n"
)


# ── Stages ───────────────────────────────────────────────────────────────────
def run_scout_stage(job: Job) -> None:
    from .clients.scout_client import MAX_REVISIONS, run_scout

    params = job.params_dict
    search = bool(params.get("search"))
    # A retry under "auto" chooses again, rather than inheriting the last choice.
    name = params.get("pathway_request") or params.get("pathway")

    from pathlib import Path

    from seesaw.pathways import catalog

    models, other = job.models, params.get("other_model")
    if not name:
        plan_path = run_scout(job.question, thread_id=f"job-{job.id}-scout",
                              output_dir=jobs.job_dir(job.id), search=search,
                              brief=catalog.models_brief(models),
                              check=lambda path: catalog.plan_model_errors(path.read_text(), models))
        jobs.update(job.id, plan_path=str(plan_path))
        return

    if name == catalog.AUTO:
        _scout_chooses(job, params, search)
        return

    def check(path: Path) -> list[str]:
        text = path.read_text()
        return (catalog.extract_inputs(name, text, models, other)[1]
                + catalog.plan_model_errors(text, models))

    plan_path = run_scout(job.question, thread_id=f"job-{job.id}-scout",
                          output_dir=jobs.job_dir(job.id), search=search,
                          brief=catalog.scout_brief(name, models, other) + "\n\n"
                          + catalog.models_brief(models), check=check)
    inputs, errors = catalog.extract_inputs(name, plan_path.read_text(), models, other)
    # Inputs from an earlier Scout run don't belong to this plan.
    params.pop("pathway_params", None)
    if errors:
        jobs.update(job.id, plan_path=str(plan_path), params=json.dumps(params))
        raise RuntimeError(f"Scout's {name} inputs are still invalid after "
                           f"{MAX_REVISIONS} revisions: " + "; ".join(errors))
    params["pathway_params"] = inputs
    jobs.update(job.id, plan_path=str(plan_path), params=json.dumps(params))
    print(f"   {name} inputs validated: {len(inputs['prompts'])} prompts")


def _scout_chooses(job: Job, params: dict, search: bool) -> None:
    """Scout picks the pathway (or none) and its inputs; the choice is written into params."""
    from pathlib import Path

    from seesaw.pathways import catalog

    from .clients.scout_client import MAX_REVISIONS, run_scout

    models, other = job.models, params.get("other_model")

    def check(path: Path) -> list[str]:
        text = path.read_text()
        return catalog.extract_choice(text, models, other)[3] + catalog.plan_model_errors(text, models)

    plan_path = run_scout(job.question, thread_id=f"job-{job.id}-scout",
                          output_dir=jobs.job_dir(job.id), search=search,
                          brief=catalog.choice_brief(models, other) + "\n\n" + catalog.models_brief(models),
                          check=check)
    name, inputs, why, errors = catalog.extract_choice(plan_path.read_text(), models, other)
    # Whatever an earlier Scout run chose doesn't belong to this plan.
    for key in ("pathway", "pathway_params", "pathway_why"):
        params.pop(key, None)
    params["pathway_request"] = catalog.AUTO
    if errors:
        jobs.update(job.id, plan_path=str(plan_path), params=json.dumps(params))
        raise RuntimeError(f"Scout's pathway choice is still invalid after {MAX_REVISIONS} "
                           "revisions: " + "; ".join(errors))
    params["pathway_why"] = why
    if name:
        params.update(pathway=name, pathway_params=inputs)
    jobs.update(job.id, plan_path=str(plan_path), params=json.dumps(params))
    events.emit(events.CHECK, "pathway choice", status=events.OK,
                summary=(f"Scout chose {name}: {why}" if name else f"Scout chose no pathway: {why}"),
                data={"pathway": name, "why": why})
    print(f"   Scout chose {name or 'no pathway'}: {why}")


def run_lens_stage(job: Job) -> None:
    from .clients.lens_client import run_lens

    if not job.plan_path:
        raise RuntimeError("no research plan on this job — Scout stage did not save one")

    from pathlib import Path

    plan_text = Path(job.plan_path).read_text()
    params = job.params_dict
    if params.get("pathway") or params.get("pathway_request"):
        from seesaw.pathways import catalog

        # Lens gets the plan without the pathway's inputs (or Scout's choice of
        # none), so it can't take them for one more experiment.
        plan_text = catalog.without_inputs(plan_text)
    # Read now, not at Scout: the user may have changed them at the gate.
    models = job.models
    plan_text += _MODELS_NOTE.format(models=", ".join(models), first=models[0])

    # A pathway job runs its pathway first, on exactly the approved inputs:
    # once per model (model_diffing once: the first model against other_model).
    pathways: list[dict] = []
    if params.get("pathway"):
        if "pathway_params" not in params:
            raise RuntimeError("this job's pathway has no approved inputs — "
                               "Scout stage did not save them")
        from seesaw.pathways import catalog

        name, inputs = params["pathway"], params["pathway_params"]
        if name == "model_diffing" and params.get("other_model"):
            inputs = {**inputs, "other_model": params["other_model"]}
        # A rerun or the gate may have changed the models; refuse inputs that
        # can't run on them here, with the reason, rather than inside the pathway.
        problems = catalog.check_models(name, inputs, models)
        if problems:
            raise RuntimeError(f"the approved {name} inputs can't run on "
                               f"{', '.join(models)}: " + "; ".join(problems))
        runs_on = models[:1] if name == "model_diffing" else models
        pathways = [_run_pathway_once(job, name, inputs, model, many=len(runs_on) > 1)
                    for model in runs_on]
        verdicts = (f"`{pathways[0]['verdict']}`" if len(pathways) == 1 else
                    ", ".join(f"`{p['verdict']}` on {p['model']}" for p in pathways))
        plan_text += _PATHWAY_DONE.format(name=name, version=pathways[0]["version"], verdict=verdicts)

    replay = _replay_of(job) if params.get("rerun_of") else None
    if replay is not None and not replay["experiments"]:
        # The source ran no extras: nothing for Lens (or its LLM) to do.
        bundle_path = jobs.job_dir(job.id) / "results_bundle.json"
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {"research_question": replay["research_question"], "model_name": models[0],
                  "results": [], "n_total": 0, "n_success": 0, "n_failed": 0}
    else:
        bundle, bundle_path = run_lens(
            plan_text,
            thread_id=f"job-{job.id}-lens",
            output_dir=jobs.job_dir(job.id),
            **({"replay": replay} if replay else {}),
        )
    if replay is not None:
        bundle["replayed_from"] = params["rerun_of"]
    if pathways:
        bundle = _with_pathways(bundle, pathways)
    if pathways or replay is not None:
        bundle_path.write_text(json.dumps(bundle, indent=2, default=str))
    # What actually ran, beside the user's choice (params["models"], never changed here).
    ran = [r.get("model_name") for r in bundle.get("results", []) if r.get("model_name")]
    params = {**jobs.get(job.id).params_dict, "models_run": list(dict.fromkeys(ran))}
    jobs.update(job.id, bundle_path=str(bundle_path), params=json.dumps(params))
    print(f"   {bundle['n_success']}/{bundle['n_total']} experiments succeeded")
    # Nothing for Quill to critique: fail here, where retry re-runs Lens,
    # rather than parking the job at a gate in front of an empty bundle.
    # (A pathway job always has the pathway's results to critique.)
    if bundle["n_total"] and not bundle["n_success"]:
        first = next((r.get("error") for r in bundle.get("results", []) if r.get("error")), None)
        raise RuntimeError(f"no experiment succeeded ({bundle['n_total']} failed)"
                           + (f"; first error: {first}" if first else ""))


def run_quill_stage(job: Job) -> None:
    from pathlib import Path

    from .clients.quill_client import run_quill

    if not job.bundle_path:
        raise RuntimeError("no results bundle on this job — Lens stage produced none")

    report, report_path = run_quill(Path(job.bundle_path), thread_id=f"job-{job.id}-quill",
                                    output_dir=jobs.job_dir(job.id))
    jobs.update(job.id, report_path=str(report_path))
    print(f"   assessment: {report.overall_assessment}")


def _replay_of(job: Job) -> dict | None:
    """The extra experiments a rerun's source job ran, to run again as they ran.

    Read from the source's results bundle, with the rerun's model swapped in.
    None when that bundle isn't here (the source never finished Lens, or ran
    on another machine): Lens then reads the plan again, and says so.
    """
    from pathlib import Path

    source_id = job.params_dict["rerun_of"]
    source = jobs.get(source_id)
    path = Path(source.bundle_path) if source and source.bundle_path else None
    if path is None or not path.exists():
        why = f"job {source_id} has no results bundle here; Lens reads the plan again instead"
        print(f"   ↳ {why}")
        events.emit(events.NOTE, "replay", status=events.FAILED, summary=why)
        return None

    bundle = json.loads(path.read_text())
    # Each experiment on the model it ran on, when the rerun keeps that model;
    # on every one of the rerun's models when it doesn't.
    models = job.models
    experiments = [
        {"name": r["name"], "tool": r["tool"], "model_name": model,
         "prompts": r["prompts"], "tool_kwargs": r.get("tool_kwargs") or {}}
        for r in bundle.get("results", []) if r.get("role") != "pathway"
        for model in ([r["model_name"]] if r.get("model_name") in models else models)
    ]
    print(f"   replaying {len(experiments)} experiment(s) from job {source_id}")
    events.emit(events.NOTE, "replay",
                summary=f"replaying {len(experiments)} experiment(s) from job {source_id} "
                        f"on {', '.join(models)}, as they ran",
                data={"source": source_id, "experiments": experiments})
    return {"research_question": bundle.get("research_question") or job.question,
            "model_name": models[0], "experiments": experiments}


def _pathway_dir(job: Job, model: str, many: bool):
    """pathway/, or pathway/<model>/ when the pathway runs on several models."""
    out = jobs.job_dir(job.id) / "pathway"
    return out / model.replace("/", "__") if many else out


def _run_pathway(job: Job, name: str, params: dict, model: str, many: bool = False) -> dict:
    """Run a pathway on one model into the job's pathway folder; return what it saved.

    pathway/result.json carries the name, version, model and inputs next to
    the verdict, so a reader (or a later retry) knows exactly what produced it.
    """
    from seesaw.pathways import catalog

    out = _pathway_dir(job, model, many)
    out.mkdir(parents=True, exist_ok=True)

    from seesaw.pathways.result import listening

    print(f"   pathway {name} on {model}")
    with events.span(events.PATHWAY, name, data={"version": catalog.version(name),
                                                 "model": model, "params": params}) as s, \
            listening(_record_step):
        result = catalog.run_from_params(name, model, params, out)
        s.summary = f"verdict: {result.verdict}" + (
            f" (stopped early: {result.stopped_early})" if result.stopped_early else "")
        s.data.update(verdict=result.verdict, evidence=result.evidence)

    (out / "report.md").write_text(result.to_markdown())
    saved = {
        "pathway": result.pathway,
        "version": catalog.version(name),
        "model": model,
        "params": params,
        "verdict": result.verdict,
        "evidence": result.evidence,
        "findings": result.findings,
        "stopped_early": result.stopped_early,
        "steps": list(result.steps),
        "plot_paths": [str(p) for p in result.plot_paths],
    }
    # Round-trip so the caller gets the same plain values a reload would.
    text = json.dumps(saved, indent=2, default=_jsonable)
    (out / "result.json").write_text(text)
    print(f"   verdict: {result.verdict}")
    return json.loads(text)


def _run_pathway_once(job: Job, name: str, params: dict, model: str, many: bool = False) -> dict:
    """The pathway's saved result when it matches, else a fresh run.

    A retried Lens stage would otherwise recompute a deterministic result it
    already has. It only counts when it came from this pathway, at this
    version, on this model and these exact inputs.
    """
    from seesaw.pathways import catalog

    path = _pathway_dir(job, model, many) / "result.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if (saved.get("pathway"), saved.get("version"), saved.get("model"), saved.get("params")) == (
                name, catalog.version(name), model, params):
            print(f"   reusing saved {name} result on {model} (verdict: {saved['verdict']})")
            events.emit(events.PATHWAY, name, summary=f"reused the saved result "
                        f"(verdict: {saved['verdict']}): same pathway, version, model and inputs")
            return saved
    return _run_pathway(job, name, params, model, many)


def _record_step(pathway: str, step: str, result, seconds: float) -> None:
    """Put one pathway step on the job's timeline as it finishes."""
    data = getattr(result, "data", None) or {}
    plots = [str(p) for p in getattr(result, "plot_paths", [])]
    scalars = [f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
               for k, v in data.items() if isinstance(v, (int, float, str)) and not isinstance(v, bool)
               and len(str(v)) < 40][:4]
    events.emit(events.STEP, step, duration_ms=seconds * 1000,
                summary=", ".join(scalars + ([f"{len(plots)} plot(s)"] if plots else [])) or None,
                data={"pathway": pathway, "data": data, "plot_paths": plots})


def _with_pathways(bundle: dict, pathways: list[dict]) -> dict:
    """Put the pathway's results (one per model) first in Lens's bundle, as the primary evidence.

    bundle["pathways"] lists them all; bundle["pathway"] is the first, for
    readers that know only one.
    """
    entries = [_pathway_entry(p) for p in pathways]
    results = entries + [{**r, "role": "extra"} for r in bundle.get("results", [])]
    return {
        **bundle,
        "pathway": pathways[0],
        "pathways": pathways,
        "results": results,
        "n_total": len(results),
        "n_success": sum(1 for r in results if r["status"] == "success"),
        "n_failed": sum(1 for r in results if r["status"] != "success"),
    }


def _pathway_entry(pathway: dict) -> dict:
    """One pathway result, in the shape of a Lens experiment result.

    Quill reads it like any other, with `role` telling the two apart:
    "pathway" for this one, "extra" for everything Lens chose to run.
    """
    return {
        "name": f"pathway:{pathway['pathway']}",
        "tool": f"{pathway['pathway']}@v{pathway['version']}",
        "role": "pathway",
        "model_name": pathway["model"],
        "prompts": pathway["params"].get("prompts", []),
        "summary": f"Verdict: {pathway['verdict']}. " + " ".join(pathway["evidence"]),
        "plot_paths": pathway["plot_paths"],
        "data": {"verdict": pathway["verdict"], "stopped_early": pathway["stopped_early"],
                 "steps": pathway["steps"], **pathway["findings"]},
        "status": "success",
        "error": None,
        "tool_kwargs": pathway["params"],
    }


def _jsonable(value):
    """Findings hold numpy scalars, tensors and paths; JSON gets plain values."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


STAGE_FNS = {
    "scout":   run_scout_stage,
    "lens":    run_lens_stage,
    "quill":   run_quill_stage,
}


# ── Driver ───────────────────────────────────────────────────────────────────
def run_job(job_id: str, worker_id: str | None = None) -> int:
    """Run queued stages for this job until it blocks, finishes, or fails.

    Returns:
        Process exit code — 0 unless a stage raised.
    """
    worker_id = worker_id or f"pid-{os.getpid()}"
    while True:
        job = jobs.claim(job_id, worker_id)
        if job is None:
            current = jobs.get(job_id)
            if current is None:
                print(f"job {job_id} not found", file=sys.stderr)
                return 1
            # Awaiting approval, cancelled, done, or another worker got it.
            print(f"job {job_id} is {current.status} at stage {current.stage} — nothing to run")
            return 0

        with events.bind(job_id, job.stage):
            code = _run_stage(job, worker_id)
        if code is not None:
            return code


def _run_stage(job: Job, worker_id: str) -> int | None:
    """Run the claimed stage. Returns an exit code, or None to run the next one."""
    job_id = job.id
    if jobs.cancel_requested(job_id):
        print("job cancelled before stage started")
        jobs.update(job_id, status=jobs.CANCELLED, pid=None)
        events.emit(events.STATUS, jobs.CANCELLED, summary="cancelled before the stage started")
        return 0

    print(f"\n▶ stage: {job.stage}")
    try:
        with events.span(events.STAGE, job.stage, data={"worker": worker_id, "model": job.model}):
            run_stage = STAGE_FNS.get(job.stage)
            if run_stage is None:
                # A standalone pathway job, queued before they were retired.
                raise RuntimeError(f"stage {job.stage!r} is no longer run: a pathway runs inside "
                                   "a research job; start one that names it, or rerun one")
            run_stage(job)
    except Exception as exc:                         # noqa: BLE001 — reported to the UI
        traceback.print_exc()
        events.emit(events.ERROR, type(exc).__name__, status=events.FAILED,
                    summary=str(exc), data={"traceback": traceback.format_exc()})
        _record_cost(job)
        jobs.update(job.id, status=jobs.FAILED,
                    error=f"{job.stage}: {type(exc).__name__}: {exc}"[:1000],
                    pid=None)
        _save_cost(job_id, next_stage=None)
        jobs.sync_log(job_id)
        return 1
    _record_cost(job)

    # Re-read: the stage wrote artifact paths, and the user may have
    # asked to cancel while it ran.
    job = jobs.get(job_id)
    if job is None or job.status == jobs.CANCELLED or job.cancel_requested:
        print("job cancelled during stage")
        jobs.update(job_id, status=jobs.CANCELLED, pid=None)
        events.emit(events.STATUS, jobs.CANCELLED, summary="cancelled while the stage ran")
        jobs.sync_log(job_id)
        return 0

    jobs.advance(job)
    after = jobs.get(job_id)
    print(f"✔ stage complete — now {after.status} at {after.stage}")
    if after.status in (jobs.AWAITING, jobs.DONE):
        cost = _save_cost(job_id, next_stage=after.stage if after.status == jobs.AWAITING else None)
        if cost:
            from shared import costs

            for line in costs.describe(cost):
                print(f"   {line}")
    events.emit(events.STATUS, after.status, summary={
        jobs.AWAITING: f"waiting for approval before {after.stage}",
        jobs.QUEUED:   f"auto-approved; {after.stage} is next",
        jobs.DONE:     "every stage finished",
    }.get(after.status, after.status))
    jobs.sync_log(job_id)

    if after.status != jobs.QUEUED:
        jobs.update(job_id, pid=None)
        return 0
    return None


def _record_cost(job: Job) -> None:
    """Put what the stage just run cost on the timeline, as the record later estimates read."""
    try:
        from shared import costs

        record = costs.stage_cost(events.list_events(job.id), job.stage)
        events.emit(events.COST, f"{job.stage} cost", data={**record, "model": job.model},
                    summary=f"{costs.usd(record['total_usd'])}: Claude {costs.usd(record['llm_usd'])}, "
                            f"GPU {costs.usd(record['gpu_usd'])}, runner {costs.usd(record['runner_usd'])}")
    except Exception as exc:                          # noqa: BLE001 — costing never fails a job
        print(f"⚠️  couldn't cost the {job.stage} stage: {type(exc).__name__}: {exc}")


def _save_cost(job_id: str, next_stage: str | None) -> dict | None:
    """Store spent-so-far and the estimate for `next_stage` on the job, for whoever approves it."""
    try:
        from shared import costs

        job = jobs.get(job_id)
        kwargs = {}
        if next_stage == "lens":
            params = job.params_dict
            kwargs["pathway"] = params.get("pathway")
            kwargs["models"] = job.models
            if job.plan_path and os.path.exists(job.plan_path):
                with open(job.plan_path) as f:
                    kwargs["plan_text"] = f.read()
        cost = costs.summary(events.list_events(job_id), job.stages, next_stage,
                             events.recent_data(events.COST, limit=200), **kwargs)
        jobs.update(job_id, cost=json.dumps(cost))
        return cost
    except Exception as exc:                          # noqa: BLE001 — costing never fails a job
        print(f"⚠️  couldn't estimate the job's cost: {type(exc).__name__}: {exc}")
        return None


def poll(worker_id: str | None = None, interval: float = 5.0) -> int:
    """Claim and run queued jobs forever.

    This is the backend's entry point once the frontend is deployed somewhere
    that can't spawn processes: Streamlit inserts a queued row, this loop
    picks it up. Safe to run several of these — claims are atomic, so two
    pollers never take the same job.
    """
    worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}"
    print(f"👷 poller {worker_id} started — checking every {interval}s")
    while True:
        # Peek rather than claim: run_job does the atomic claim, and that's
        # what decides the winner when several pollers see the same row.
        job_id = jobs.next_queued_id()
        if job_id is None:
            time.sleep(interval)
            continue
        job = jobs.get(job_id)
        print(f"\n📥 picking up job {job_id} at stage {job.stage}: {job.question[:60]}")
        try:
            run_job(job_id, worker_id=worker_id)
        except Exception:                            # noqa: BLE001 — keep polling
            traceback.print_exc()


def main() -> int:
    load_dotenv()
    args = sys.argv[1:]
    if args == ["--poll"]:
        return poll()
    if len(args) != 1:
        print("usage: python -m orchestrator.src.worker <job_id> | --poll",
              file=sys.stderr)
        return 2
    return run_job(args[0])


if __name__ == "__main__":
    raise SystemExit(main())
