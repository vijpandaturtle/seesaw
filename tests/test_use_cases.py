"""The use-case catalog: what Seesaw is for, and how a job records which use case it serves.

The catalog has to stay consistent with the pathways and lever tools it names,
and a job for a use case runs that use case's pathway and keeps the use case on
a rerun. The dashboard keeps a copy of the catalog's JSON, so the JSON's shape
is checked too.
"""

from __future__ import annotations

import json
import sys

import pytest

from seesaw import use_cases
from seesaw.pathways import catalog


def test_the_catalog_is_consistent():
    assert use_cases.problems() == []


def test_every_offered_pathway_belongs_to_a_use_case():
    offered = [p for p in catalog.names() if p != "attribute_audit"]
    assert {p: use_cases.for_pathway(p) for p in offered if use_cases.for_pathway(p) is None} == {}
    assert use_cases.for_pathway("attribute_audit") == use_cases.for_pathway("concept_direction")
    assert use_cases.for_pathway(None) is None


def test_a_pathway_maps_to_the_use_case_that_lists_it_first():
    # factual_recall is a supporting pathway of edit_or_unlearn_fact, but locate_a_fact leads with it.
    assert use_cases.for_pathway("factual_recall") == "locate_a_fact"
    assert use_cases.for_pathway("editing_comparison") == "unlearn_knowledge"


def test_problems_catches_a_bad_entry(monkeypatch):
    bad = dict(use_cases.USE_CASES)
    bad["broken"] = use_cases.UseCase("Broken", "?", "m", "b", status="runnable", area="fix", pathways=("nope",))
    bad["vague"] = use_cases.UseCase("Vague", "?", "m", "b", status="partial", area="monitor", pathways=("probe_monitor",))
    monkeypatch.setattr(use_cases, "USE_CASES", bad)
    found = use_cases.problems()
    assert "broken: unknown pathway nope" in found
    assert any(f.startswith("vague: a gap") for f in found)


def test_json_is_what_the_dashboard_reads(tmp_path):
    out = use_cases.write_json(tmp_path / "use-cases.json")
    data = json.loads(out.read_text())
    assert data["statuses"] == ["runnable", "partial", "planned"]
    first = data["use_cases"][0]
    assert set(first) == {"name", *use_cases.UseCase._fields}
    assert isinstance(first["pathways"], list)
    assert [c["name"] for c in data["use_cases"]] == use_cases.names()


# ── Jobs ─────────────────────────────────────────────────────────────────────
@pytest.fixture
def jobs(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("MODAL_RUN_STAGE_URL", raising=False)
    monkeypatch.setenv("SEESAW_DB_PATH", str(tmp_path / "jobs.db"))
    from shared.db import jobs as module

    monkeypatch.setattr(module, "LOGS_DIR", tmp_path / "job_logs")
    return module


def seesaw(monkeypatch, *argv) -> int:
    from seesaw import cli

    monkeypatch.setattr(cli, "_load_env", lambda: None)
    monkeypatch.setattr(sys, "argv", ["seesaw", *argv])
    return cli.main()


def test_a_use_case_runs_its_main_pathway(jobs):
    job = jobs.create("q", params={"use_case": "remove_spurious_correlation"})
    assert job.params_dict["pathway"] == "spurious_audit"
    assert job.params_dict["use_case"] == "remove_spurious_correlation"


def test_a_use_case_can_name_another_of_its_pathways(jobs):
    job = jobs.create("q", params={"use_case": "unlearn_knowledge", "pathway": "factual_recall"})
    assert job.params_dict["pathway"] == "factual_recall"


@pytest.mark.parametrize("params, message", [
    ({"use_case": "nope"}, "unknown use case 'nope'"),
    ({"use_case": "find_a_backdoor"}, "is planned, not runnable yet"),
    ({"use_case": "build_a_safety_monitor", "pathway": "circuit_discovery"}, "runs probe_monitor"),
])
def test_a_use_case_job_is_checked(jobs, params, message):
    with pytest.raises(ValueError, match=message):
        jobs.create("q", params=params)


def test_a_rerun_keeps_the_use_case(jobs, monkeypatch, tmp_path):
    monkeypatch.setattr(catalog, "_tokenizer", lambda model: None)
    inputs = {"prompts": ["When Mary and John went to the store, John gave a drink to"],
              "positive_tokens": [" Mary"], "negative_tokens": [" John"]}
    job = jobs.create("IOI", params={"use_case": "understand_a_capability"})
    jobs.update(job.id, stage="quill", plan_path=str(tmp_path / "plan.md"),
                params=json.dumps({**job.params_dict, "pathway_params": inputs}))
    again = jobs.rerun(job.id)
    assert again.params_dict["use_case"] == "understand_a_capability"


def test_submit_takes_a_use_case(jobs, monkeypatch, capsys):
    argv = ("jobs", "submit", "q", "--use-case", "build_a_safety_monitor", "--runner", "queue")
    assert seesaw(monkeypatch, *argv) == 0
    [job] = jobs.list_jobs()
    assert job.params_dict == {"use_case": "build_a_safety_monitor", "pathway": "probe_monitor",
                               "models": [jobs.DEFAULT_MODEL]}


def test_use_cases_command_lists_and_writes_json(monkeypatch, capsys, tmp_path):
    assert seesaw(monkeypatch, "use-cases") == 0
    out = capsys.readouterr().out
    assert "remove_spurious_correlation" in out and "baseline:" in out
    path = tmp_path / "out.json"
    assert seesaw(monkeypatch, "use-cases", "--json", str(path)) == 0
    assert json.loads(path.read_text())["use_cases"]


def test_every_use_case_has_a_safety_area_and_the_json_lists_them(tmp_path):
    data = use_cases.as_json()
    assert [a["name"] for a in data["areas"]] == ["monitor", "audit", "fix", "understand"]
    assert {c["area"] for c in data["use_cases"]} <= set(use_cases.AREAS)
    # Every area but understand has something in it that's a job with an outcome to beat.
    for area in ("monitor", "audit", "fix"):
        assert any(c.area == area for c in use_cases.USE_CASES.values()), area
