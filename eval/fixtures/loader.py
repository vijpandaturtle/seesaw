"""Grader-validation fixtures: canned (inputs, expected verdict) pairs per grader.

These validate that the GRADERS discriminate — the meta-eval. Each fixture pins one
grader to a set of inputs and the verdict it must produce, with a balanced pass AND
fail case per grader (the one-sided-eval guard, applied to the graders themselves).

Fixture file shape (a `fixtures:` list per YAML file):

    fixtures:
      - id: "plan_well_formed_pass"
        grader: scout.plan_well_formed      # <agent>.<function> in eval/graders/
        expected: pass                      # pass | fail | na
        inputs: {plan: "## Hypothesis ..."} # kwargs handed to the grader
        tags: []                            # [] = deterministic/offline-runnable;
                                            # [llm] needs the judge; [network] hits the net

Verdict bands (chosen so a borderline LLM-judge call can't flap the test):
    pass -> score >= PASS_BAND (0.6)
    fail -> score <= FAIL_BAND (0.4)
    na   -> score is None (grader not applicable / needs a human)
A score landing in the 0.4–0.6 dead-band is INDETERMINATE — signals a weak grader
or a badly-chosen fixture, and fails the meta-eval on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from eval.graders import lens, quill, scout
from eval.graders.base import GraderResult

FIXTURES_DIR = Path(__file__).resolve().parent
PASS_BAND = 0.6
FAIL_BAND = 0.4

_GRADER_MODULES = {"scout": scout, "lens": lens, "quill": quill}


@dataclass
class Fixture:
    id: str
    grader: str                       # "<agent>.<func>"
    expected: str                     # pass | fail | na
    inputs: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    note: str = ""
    file_path: Path | None = None

    @property
    def agent(self) -> str:
        return self.grader.split(".", 1)[0]

    @property
    def offline(self) -> bool:
        """Runnable now with no API key and no network."""
        return not ({"llm", "network"} & set(self.tags))


class FixtureError(ValueError):
    """Bad fixture definition — unknown grader, bad expected, etc."""


def resolve_grader(dotted: str):
    """'scout.plan_well_formed' -> the callable in eval/graders/scout.py."""
    try:
        agent, func = dotted.split(".", 1)
    except ValueError:
        raise FixtureError(f"grader {dotted!r} must be '<agent>.<func>'")
    mod = _GRADER_MODULES.get(agent)
    if mod is None or not hasattr(mod, func):
        raise FixtureError(f"unknown grader {dotted!r}")
    return getattr(mod, func)


def band(score: float | None) -> str:
    """Map a GraderResult.score to a verdict band."""
    if score is None:
        return "na"
    if score >= PASS_BAND:
        return "pass"
    if score <= FAIL_BAND:
        return "fail"
    return "indeterminate"


def load_all(directory: str | Path = FIXTURES_DIR) -> list[Fixture]:
    fixtures: list[Fixture] = []
    seen: set[str] = set()
    for path in sorted(Path(directory).glob("*_fixtures.y*ml")):
        raw = yaml.safe_load(path.read_text()) or {}
        for i, fx in enumerate(raw.get("fixtures", []) or []):
            for req in ("id", "grader", "expected"):
                if req not in fx:
                    raise FixtureError(f"{path.name}[{i}]: missing {req!r}")
            if fx["expected"] not in ("pass", "fail", "na"):
                raise FixtureError(f"{path.name}[{i}] {fx['id']}: expected must be pass|fail|na")
            resolve_grader(fx["grader"])                       # validates the grader exists
            if fx["id"] in seen:
                raise FixtureError(f"duplicate fixture id {fx['id']!r}")
            seen.add(fx["id"])
            fixtures.append(Fixture(
                id=fx["id"], grader=fx["grader"], expected=fx["expected"],
                inputs=fx.get("inputs", {}) or {}, tags=list(fx.get("tags", []) or []),
                note=fx.get("note", ""), file_path=path,
            ))
    return fixtures


def run_fixture(fx: Fixture) -> tuple[str, GraderResult]:
    """Execute the grader on the fixture's inputs; return (actual_band, result)."""
    res = resolve_grader(fx.grader)(**fx.inputs)
    return band(res.score), res


def check_llm(fixtures: list[Fixture] | None = None, *, agent: str | None = None) -> dict[str, Any]:
    """Run the [llm]-tagged fixtures through the real judge and compare bands.

    This is the JUDGE meta-eval: a mismatch means either the judge or the fixture
    label is wrong — read the detail before deciding which. Costs API credits
    (one judge call per fixture); filter with `agent=` to run one batch at a time.
    Requires ANTHROPIC_API_KEY. Network/offline fixtures are skipped here.
    """
    fixtures = fixtures if fixtures is not None else load_all()
    targets = [f for f in fixtures
               if "llm" in f.tags and (agent is None or f.agent == agent)]
    ran, matches, results = 0, 0, []
    for fx in targets:
        try:
            actual, res = run_fixture(fx)
            err = None
        except Exception as exc:  # noqa: BLE001 — one judge failure shouldn't kill the batch
            actual, res, err = "error", None, repr(exc)
        ran += 1
        ok = actual == fx.expected
        matches += ok
        results.append({
            "id": fx.id, "grader": fx.grader, "expected": fx.expected,
            "actual": actual, "ok": ok,
            "score": res.score if res else None,
            "detail": (res.detail if res else err or "")[:200],
        })
    return {"ran": ran, "matched": matches, "results": results}


def human_review_payloads(fixtures: list[Fixture] | None = None,
                          *, agent: str | None = None) -> list[dict[str, Any]]:
    """Execute the human graders and collect their packaged review payloads.

    Human graders don't score — they format what an expert needs to see
    (artifact + rubric). This gathers those payloads so they can be written to a
    file / doc for actual human review.
    """
    fixtures = fixtures if fixtures is not None else load_all()
    payloads = []
    for fx in fixtures:
        if fx.expected != "na" or (agent and fx.agent != agent):
            continue
        _, res = run_fixture(fx)
        if res.needs_human:
            payloads.append({"fixture_id": fx.id, "grader": fx.grader,
                             "instructions": res.detail,
                             **res.metrics.get("review_payload", {})})
    return payloads


def check_offline(fixtures: list[Fixture] | None = None) -> dict[str, Any]:
    """Run every offline (deterministic) fixture and report matches/mismatches.

    Skips [llm]/[network] fixtures (they need an API key / the net). Returns a
    summary with mismatches (actual band != expected) and per-grader pass/fail
    coverage so you can see every criterion has both cases.
    """
    fixtures = fixtures if fixtures is not None else load_all()
    ran, mismatches = 0, []
    coverage: dict[str, set[str]] = {}
    for fx in fixtures:
        coverage.setdefault(fx.grader, set()).add(fx.expected)
        if not fx.offline:
            continue
        actual, res = run_fixture(fx)
        ran += 1
        if actual != fx.expected:
            mismatches.append({"id": fx.id, "grader": fx.grader,
                               "expected": fx.expected, "actual": actual,
                               "score": res.score, "detail": res.detail})
    return {
        "total": len(fixtures), "ran_offline": ran, "mismatches": mismatches,
        "coverage": {g: sorted(v) for g, v in sorted(coverage.items())},
    }
