"""Fixtures for the deterministic eval tests.

These tests check what the agents actually produced against findings that are
already established in the literature — L9H9 as an IOI name mover, mid-layer
MLPs carrying factual recall, and so on. They assert on the run's artifacts
rather than re-running any model, so they're fast and free.

The runner writes the artifact paths into the environment before invoking
pytest (see eval/runner/handlers.py::_artifact_env). Running these directly is
possible too:

    SEESAW_EVAL_BUNDLE=outputs/results_bundle.json pytest eval/tests -q

A missing artifact fails rather than skips: if the pipeline was supposed to
produce a bundle and didn't, that's a result about the pipeline, not a reason
to stay quiet.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    path = Path(raw)
    return path if path.exists() else None


@pytest.fixture(scope="session")
def bundle() -> dict:
    """The ExperimentBundle this run produced."""
    path = _env_path("SEESAW_EVAL_BUNDLE")
    if path is None:
        pytest.fail(
            "no experiment bundle for this run — set SEESAW_EVAL_BUNDLE, or the "
            "pipeline produced nothing for these tests to check"
        )
    return json.loads(path.read_text())


@pytest.fixture(scope="session")
def plan_text() -> str:
    path = _env_path("SEESAW_EVAL_PLAN")
    if path is None:
        pytest.fail("no research plan for this run — set SEESAW_EVAL_PLAN")
    return path.read_text()


@pytest.fixture(scope="session")
def results(bundle: dict) -> list[dict]:
    """Successful experiment results only."""
    return [r for r in bundle.get("results", []) if r.get("status") == "success"]


@pytest.fixture(scope="session")
def by_tool(results: list[dict]) -> dict[str, list[dict]]:
    """Successful results grouped by the tool that produced them."""
    grouped: dict[str, list[dict]] = {}
    for r in results:
        grouped.setdefault(r.get("tool", "unknown"), []).append(r)
    return grouped


# ── helpers ──────────────────────────────────────────────────────────────────
def require_tool(by_tool: dict[str, list[dict]], tool: str) -> list[dict]:
    """Results for `tool`, failing with a useful message when there are none.

    An absent tool means the agent never ran the experiment this test is
    about, so there is nothing to check — that's a failure of the run.
    """
    hits = by_tool.get(tool)
    if not hits:
        ran = sorted(by_tool) or ["(nothing succeeded)"]
        pytest.fail(f"no successful {tool!r} experiment in this run; got {ran}")
    return hits


def head_names(result: dict) -> list[str]:
    """['L9H9', 'L10H7', ...] from a tool that reports top_heads."""
    return [name for name, _score in result.get("data", {}).get("top_heads", [])]


def head_scores(result: dict) -> dict[str, float]:
    return {name: score for name, score in result.get("data", {}).get("top_heads", [])}


def layer_of(head: str) -> int:
    """'L9H9' -> 9"""
    return int(head.split("H")[0].lstrip("L"))
