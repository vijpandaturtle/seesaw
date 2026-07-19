"""Handlers for the five task grader types. Each takes (GraderSpec, Task,
AgentRun) and returns list[GraderResult].

Design rule: a handler must never silently pass. A missing test file, an
unimplemented state_check key, or an absent artifact is reported as an explicit
error/needs_human result — visible in the TaskResult, not swallowed.
"""

from __future__ import annotations

import fnmatch
import subprocess
import sys
from pathlib import Path

from ..graders import GraderResult, GraderType, field as read
from ..graders.base import judge
from ..tasks import GraderSpec, Task
from .core import AgentRun

EVAL_DIR = Path(__file__).resolve().parents[1]
TESTS_DIR = EVAL_DIR / "tests"
REPO_ROOT = EVAL_DIR.parent

# Mirrors lens TOOL_REGISTRY (kept flat to avoid the torch import).
LENS_TOOLS = {"logit_lens", "attention_pattern", "direct_logit_attribution",
              "ablation", "activation_patching"}


def _res(criterion: str, agent: str, kind: GraderType, **kw) -> GraderResult:
    return GraderResult(criterion=criterion, agent=agent, grader_type=kind, **kw)


# ── deterministic_tests ──────────────────────────────────────────────────────
def handle_deterministic_tests(spec: GraderSpec, task: Task, run: AgentRun) -> list[GraderResult]:
    out = []
    for test_file in spec.config["required"]:
        path = TESTS_DIR / test_file
        crit = f"deterministic_tests.{Path(test_file).stem}"
        if not path.exists():
            out.append(_res(crit, task.target or "pipeline", GraderType.CODE, score=None,
                            error=f"test file not found: {path}",
                            detail="referenced test not implemented yet"))
            continue
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(path), "-x", "-q",
             "--no-header", "-p", "no:cacheprovider"],
            capture_output=True, text=True, cwd=REPO_ROOT, timeout=1800,
        )
        tail = (proc.stdout or proc.stderr).strip().splitlines()[-1] if (proc.stdout or proc.stderr) else ""
        out.append(_res(crit, task.target or "pipeline", GraderType.CODE,
                        score=1.0 if proc.returncode == 0 else 0.0,
                        metrics={"returncode": proc.returncode},
                        detail=tail[:200]))
    return out


# ── llm_rubric ───────────────────────────────────────────────────────────────
def _artifact_for(target: str, run: AgentRun) -> str | None:
    """Pick the text the rubric grades, by target agent."""
    if target == "scout":
        return run.plan_text
    if target == "lens":
        if not run.bundle:
            return None
        lines = [f"Research question: {run.bundle.get('research_question')}"]
        for r in run.bundle.get("results", []):
            lines.append(f"[{r.get('tool')}] {r.get('name')}: {r.get('summary', '')}")
        return "\n".join(lines)
    if target == "quill":
        rep = run.report
        if rep is None:
            return None
        gaps = [read(g, "description") for g in (read(rep, "gaps", []) or [])]
        bundle_txt = _artifact_for("lens", run) or str(run.bundle)[:2000]
        return (f"BUNDLE UNDER REVIEW:\n{bundle_txt}\n\nCRITIQUE:\n"
                f"assessment: {read(rep, 'overall_assessment')}\n"
                f"summary: {read(rep, 'overall_summary')}\ngaps: {gaps}")
    return None


def handle_llm_rubric(spec: GraderSpec, task: Task, run: AgentRun) -> list[GraderResult]:
    from pydantic import BaseModel

    target = spec.config.get("target") or task.target or "scout"
    crit = f"llm_rubric.{Path(spec.config['rubric']).stem}"
    rubric_path = EVAL_DIR / spec.config["rubric"]
    if not rubric_path.exists():
        return [_res(crit, target, GraderType.LLM, score=None,
                     error=f"rubric not found: {rubric_path}")]
    artifact = _artifact_for(target, run)
    if artifact is None:
        return [_res(crit, target, GraderType.LLM, score=None,
                     error=f"no artifact for target {target!r} (agent produced nothing to grade)")]

    class Grade(BaseModel):
        passed: bool
        score: float          # normalized 0-1 per the rubric's own scale
        reasoning: str

    g: "Grade" = judge(
        "Apply the following rubric to the artifact. Return passed (bool), a "
        "normalized score in [0,1] mapping the rubric's scale, and reasoning.\n\n"
        f"RUBRIC:\n{rubric_path.read_text()}\n\nARTIFACT:\n{artifact[:6000]}",
        Grade, model=spec.config.get("model") or "claude-opus-4-8",
    )
    return [_res(crit, target, GraderType.LLM, score=max(0.0, min(1.0, g.score)),
                 metrics={"passed": g.passed}, detail=g.reasoning)]


# ── static_analysis ──────────────────────────────────────────────────────────
def handle_static_analysis(spec: GraderSpec, task: Task, run: AgentRun) -> list[GraderResult]:
    out = []
    paths = spec.config.get("paths") or ["."]
    for cmd in spec.config["commands"]:
        crit = f"static_analysis.{cmd}"
        try:
            proc = subprocess.run([cmd, *paths], capture_output=True, text=True,
                                  cwd=REPO_ROOT, timeout=600)
            out.append(_res(crit, task.target or "pipeline", GraderType.CODE,
                            score=1.0 if proc.returncode == 0 else 0.0,
                            metrics={"returncode": proc.returncode},
                            detail=(proc.stdout or proc.stderr).strip()[:200]))
        except FileNotFoundError:
            out.append(_res(crit, task.target or "pipeline", GraderType.CODE, score=None,
                            error=f"{cmd} not installed"))
    return out


# ── state_check ──────────────────────────────────────────────────────────────
def handle_state_check(spec: GraderSpec, task: Task, run: AgentRun) -> list[GraderResult]:
    """Interpret the task's `expect:` mapping against the AgentRun artifacts.
    Implemented keys below; anything else is reported needs_human, never passed."""
    out = []
    agent = task.target or "pipeline"
    for artifact, expectations in spec.config["expect"].items():
        for key, expected in (expectations or {}).items():
            crit = f"state_check.{artifact}.{key}"
            checked, ok, detail = True, False, ""

            if artifact == "research_plan" and key == "saved":
                ok = (run.plan_path is not None) == bool(expected)
                detail = f"plan_path={run.plan_path}"
            elif artifact == "research_plan" and key == "target_model":
                ok = bool(run.plan_text) and str(expected).lower() in run.plan_text.lower()
                detail = f"expected {expected!r} named in plan"
            elif artifact == "experiment_bundle" and key == "schema_valid":
                required = {"research_question", "model_name", "results"}
                ok = bool(run.bundle) and required <= set(run.bundle)
                detail = f"bundle keys: {sorted(run.bundle) if run.bundle else None}"
            elif artifact == "experiment_bundle" and key == "min_successful_results":
                n = sum(1 for r in (run.bundle or {}).get("results", [])
                        if r.get("status") == "success")
                ok = n >= int(expected)
                detail = f"{n} successful results (need >={expected})"
            elif artifact == "critique_report" and key == "saved":
                ok = (run.report is not None) == bool(expected)
                detail = f"report_path={run.report_path}"
            elif artifact == "critique_report" and key == "overall_assessment_in":
                obs = read(run.report, "overall_assessment") if run.report else None
                ok = obs in list(expected)
                detail = f"observed={obs!r}, allowed={list(expected)}"
            elif artifact == "critique_report" and key == "min_followups":
                n = len(read(run.report, "followups", []) or []) if run.report else 0
                ok = n >= int(expected)
                detail = f"{n} follow-ups (need >={expected})"
            elif artifact == "followups" and key == "tools_in_registry":
                fups = read(run.report, "followups", []) or [] if run.report else []
                bad = [read(f, "tool") for f in fups if read(f, "tool") not in LENS_TOOLS]
                ok = not bad if fups else False
                detail = f"invalid tools: {bad}" if bad else f"{len(fups)} follow-ups, all registry tools"
            else:
                checked = False

            if checked:
                out.append(_res(crit, agent, GraderType.CODE, score=1.0 if ok else 0.0, detail=detail))
            else:
                out.append(_res(crit, agent, GraderType.CODE, score=None, needs_human=True,
                                detail=f"state_check key {artifact}.{key} not implemented — verify manually"))
    return out


# ── tool_calls ───────────────────────────────────────────────────────────────
def _params_match(required: dict, actual: dict) -> bool:
    """Subset match; string values support glob patterns (e.g. path: 'src/auth/*')."""
    for k, v in (required or {}).items():
        a = actual.get(k)
        if isinstance(v, str) and isinstance(a, str):
            if not fnmatch.fnmatch(a, v):
                return False
        elif a != v:
            return False
    return True


def handle_tool_calls(spec: GraderSpec, task: Task, run: AgentRun) -> list[GraderResult]:
    out = []
    agent = task.target or "pipeline"
    for req in spec.config.get("required", []) or []:
        tool, params = req.get("tool"), req.get("params", {})
        hit = any(c["name"] == tool and _params_match(params, c.get("args", {}))
                  for c in run.trajectory)
        out.append(_res(f"tool_calls.required.{tool}", agent, GraderType.CODE,
                        score=1.0 if hit else 0.0,
                        detail=f"{tool} {'called' if hit else 'NOT called'} "
                               f"({len(run.trajectory)} calls in trajectory)"))
    for forb in spec.config.get("forbidden", []) or []:
        tool, params = forb.get("tool"), forb.get("params", {})
        hit = any(c["name"] == tool and _params_match(params, c.get("args", {}))
                  for c in run.trajectory)
        out.append(_res(f"tool_calls.forbidden.{tool}", agent, GraderType.CODE,
                        score=0.0 if hit else 1.0,
                        detail=f"{tool} {'CALLED (forbidden)' if hit else 'correctly not called'}"))
    return out


# ── dispatch ─────────────────────────────────────────────────────────────────
_HANDLERS = {
    "deterministic_tests": handle_deterministic_tests,
    "llm_rubric": handle_llm_rubric,
    "static_analysis": handle_static_analysis,
    "state_check": handle_state_check,
    "tool_calls": handle_tool_calls,
}


def dispatch_grader(spec: GraderSpec, task: Task, run: AgentRun) -> list[GraderResult]:
    handler = _HANDLERS.get(spec.type)
    if handler is None:   # loader validates types, but never silently pass
        return [_res(f"unknown.{spec.type}", task.target or "pipeline", GraderType.CODE,
                     score=None, error=f"no handler for grader type {spec.type!r}")]
    try:
        return handler(spec, task, run)
    except Exception as exc:  # noqa: BLE001 — one grader must not kill the task
        return [_res(f"{spec.type}.error", task.target or "pipeline", GraderType.CODE,
                     score=None, error=repr(exc), detail="handler raised")]
