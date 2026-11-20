"""A job starts from a behavior with examples: checking it, reading it from a file, briefing Scout."""

from __future__ import annotations

import sys

import pytest

from seesaw import behavior


def test_clean_normalises_examples_and_drops_empty_parts():
    assert behavior.clean({
        "examples": ["  Are you sure? ", {"prompt": "17 x 24?", "output": " 418 "}],
        "non_examples": [],
        "action": "",
    }) == {"examples": [{"prompt": "Are you sure?"}, {"prompt": "17 x 24?", "output": "418"}]}
    assert behavior.clean(None) == {} == behavior.clean({})


@pytest.mark.parametrize("bad, match", [
    ({"exampels": []}, "takes only"),
    ({"examples": "one"}, "is a list"),
    ({"examples": [{"prompt": " "}]}, "non-empty prompt"),
    ({"examples": [{"prompt": "p", "label": 1}]}, "each example"),
    ({"examples": ["p"] * 21}, "at most 20"),
    ({"action": "deploy"}, "action is one of"),
    ({"examples": ["x" * 4001]}, "over 4000"),
])
def test_clean_rejects_what_scout_could_not_use(bad, match):
    with pytest.raises(ValueError, match=match):
        behavior.clean(bad)


def test_load_reads_yaml_and_needs_the_behavior_sentence(tmp_path):
    path = tmp_path / "b.yaml"
    path.write_text("behavior: Caves to pushback\naction: audit\nexamples:\n"
                    "  - prompt: 'Are you sure?'\n    output: 'Sorry, you are right.'\n")
    assert behavior.load(path) == ("Caves to pushback", {
        "examples": [{"prompt": "Are you sure?", "output": "Sorry, you are right."}],
        "action": "audit"})
    path.write_text("examples: [p]\n")
    with pytest.raises(ValueError, match="`behavior` is required"):
        behavior.load(path)


def test_brief_shows_the_examples_and_asks_for_a_testable_claim():
    text = behavior.brief("Caves to pushback", {
        "examples": [{"prompt": "Are you sure?", "output": "Sorry!"}],
        "non_examples": [{"prompt": "Thanks!"}],
        "action": "monitor"})
    for needed in ("Caves to pushback", "testable claim", '"Are you sure?"', 'Model said: "Sorry!"',
                   '"Thanks!"', "Match lengths", "Monitor: catch unsafe behavior at runtime",
                   "`probe_monitor`", "baseline without interpretability"):
        assert needed in text
    assert "gave no examples" in behavior.brief("Caves to pushback", None)


def test_the_action_brief_lists_only_that_areas_runnable_use_cases():
    from seesaw.use_cases import USE_CASES

    for action in behavior.ACTIONS:
        text = behavior.brief("b", {"action": action})
        for name, u in USE_CASES.items():
            listed = f"(`{name}`)" in text
            assert listed == (u.area == action and u.status != "planned"), (action, name)
    assert "no outcome to beat" in behavior.brief("b", {"action": "understand"})


# ── the store and the CLI ────────────────────────────────────────────────────
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


def test_the_store_keeps_the_behavior_and_a_rerun_copies_it(jobs):
    job = jobs.create("Caves to pushback", params={"behavior": {"examples": ["Are you sure?"]}})
    assert job.question == "Caves to pushback"
    assert job.params_dict["behavior"] == {"examples": [{"prompt": "Are you sure?"}]}
    assert "behavior" not in jobs.create("Caves", params={"behavior": {}}).params_dict
    with pytest.raises(ValueError, match="non-empty prompt"):
        jobs.create("Caves", params={"behavior": {"examples": [""]}})
    with pytest.raises(ValueError, match="what behavior to investigate"):
        jobs.create("  ")


def test_a_use_case_sets_the_action_to_its_area(jobs):
    from seesaw.use_cases import USE_CASES

    name, case = next((n, u) for n, u in USE_CASES.items() if u.status != "planned")
    job = jobs.create("Caves", params={"use_case": name})
    assert job.params_dict["behavior"] == {"action": case.area}
    other = next(a for a in behavior.ACTIONS if a != case.area)
    with pytest.raises(ValueError, match=f"is for {case.area}, not {other}"):
        jobs.create("Caves", params={"use_case": name, "behavior": {"action": other}})


def test_submit_takes_examples_from_flags_and_a_file(jobs, monkeypatch, tmp_path, capsys):
    argv = ("jobs", "submit", "Caves to pushback", "--example", "Are you sure?",
            "--non-example", "Thanks!", "--action", "audit", "--runner", "queue")
    assert seesaw(monkeypatch, *argv) == 0
    [job] = jobs.list_jobs()
    assert job.question == "Caves to pushback"
    assert job.params_dict["behavior"] == {"examples": [{"prompt": "Are you sure?"}],
                                           "non_examples": [{"prompt": "Thanks!"}],
                                           "action": "audit"}

    path = tmp_path / "b.yaml"
    path.write_text("behavior: Flatters users\nexamples:\n  - prompt: p1\n    output: o1\n")
    assert seesaw(monkeypatch, "jobs", "submit", "--behavior-file", str(path),
                  "--example", "p2", "--runner", "queue") == 0
    newest = jobs.list_jobs()[0]
    assert newest.question == "Flatters users"
    assert newest.params_dict["behavior"] == {"examples": [{"prompt": "p1", "output": "o1"},
                                                           {"prompt": "p2"}]}

    assert seesaw(monkeypatch, "jobs", "submit", "--runner", "queue") == 1
    assert "say what behavior to investigate" in capsys.readouterr().err
    assert seesaw(monkeypatch, "jobs", "submit", "--behavior-file", str(tmp_path / "nope.yaml"),
                  "--runner", "queue") == 1
