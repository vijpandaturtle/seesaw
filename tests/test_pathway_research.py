"""Research jobs that name a pathway: Scout picks its inputs, Lens runs it first.

What has to hold:
  * a research job without a pathway runs Scout exactly as before
  * with one, Scout is told the pathway's inputs, and the inputs it writes
    are checked and saved on the job for the human gate to show
  * invalid inputs go back to Scout, at most twice, then fail the stage
  * Lens's stage runs the pathway on exactly the approved inputs before
    Lens's own experiments, which are labelled extras, and a retry reuses the
    saved pathway result instead of recomputing it
  * Quill is told the pathway result is the primary evidence
  * a rerun starts at Lens with the same plan and approved inputs (no Scout),
    runs the pathway afresh, may move it to another model, and refuses
    inputs that can't run there
  * a rerun's Lens replays the source job's extra experiments as they ran,
    on the rerun's model, and reads the plan again only when the source's
    bundle isn't there

Scout's agent, Lens, Quill and the pathway itself are faked; nothing here
calls an LLM or loads a model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from seesaw.pathways import catalog
from seesaw.pathways.result import PathwayResult

IOI = {
    "prompts": ["When Mary and John went to the store, John gave a drink to"],
    "positive_tokens": [" Mary"],
    "negative_tokens": [" John"],
}


def plan(params: dict | None = IOI, pathway: str = "circuit_discovery") -> str:
    """A Scout plan, with a pathway ```json block when params is given."""
    text = "# Research Plan: IOI\n\n## Experiments\n1. logit lens on the IOI prompts\n"
    if params is not None:
        text += "\n```json\n" + json.dumps({"pathway": pathway, "params": params}) + "\n```\n"
    return text


class FakeAgent:
    """Saves the next canned plan on every turn, and records what it was sent."""

    def __init__(self, output_dir, plans):
        self.output_dir, self.plans, self.messages = output_dir, list(plans), []

    def stream(self, payload, config, stream_mode):
        self.messages.append(payload["messages"][0]["content"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "ioi_research_plan.md").write_text(self.plans.pop(0))
        return iter(())


class FakeTokenizer:
    """Words longer than six characters split in two, like a real BPE might."""

    def encode(self, text, add_special_tokens=False):
        return [0] if len(text.strip()) <= 6 else [0, 1]

    def convert_ids_to_tokens(self, ids):
        return [f"t{i}" for i in ids]


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("SEESAW_ARTIFACTS_DIR", raising=False)
    monkeypatch.setenv("SEESAW_DB_PATH", str(tmp_path / "jobs.db"))
    from shared.db import jobs as module

    monkeypatch.setattr(module, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(module, "LOGS_DIR", tmp_path / "job_logs")
    monkeypatch.setattr(catalog, "_tokenizer", lambda model: FakeTokenizer())
    return module


@pytest.fixture
def scout(jobs, monkeypatch):
    """Fake Scout's agent; call with the plans it should save, in order."""
    from orchestrator.src.clients import scout_client

    made = []

    def install(*plans):
        def build_agent(output_dir, search=True):
            made.append(FakeAgent(output_dir, plans))
            made[-1].search = search
            return made[-1]
        monkeypatch.setattr(scout_client, "build_agent", build_agent)
        return made
    return install


@pytest.fixture
def pathway_runs(monkeypatch):
    """Fake the pathway; the list records every run's inputs."""
    runs = []

    def fake_run(name, model_name, params, output_dir):
        runs.append(params)
        (plot := output_dir / "dla.png").write_bytes(b"png")
        result = PathwayResult(pathway=name, verdict="circuit_found",
                               evidence=["L9H9 carries 40% of the logit diff."],
                               findings={"baseline_logit_diff": 3.5})
        result.steps["dla"] = type("R", (), {"plot_paths": [plot]})()
        return result

    monkeypatch.setattr(catalog, "run_from_params", fake_run)
    return runs


@pytest.fixture
def lens(monkeypatch):
    """Fake Lens: one extra experiment. Records the plan text it was given."""
    from orchestrator.src.clients import lens_client

    seen = {"plans": [], "replays": [], "fail": False}

    def fake_lens(research_plan, thread_id, output_dir, replay=None):
        seen["plans"].append(research_plan)
        seen["replays"].append(replay)
        if seen["fail"]:
            raise RuntimeError("Lens fell over")
        bundle = {"research_question": "IOI", "model_name": "gpt2", "n_total": 1,
                  "n_success": 1, "n_failed": 0,
                  "results": [{"name": "logit_lens_ioi", "tool": "logit_lens",
                               "model_name": "gpt2", "prompts": IOI["prompts"],
                               "summary": "answer appears at layer 9", "status": "success"}]}
        path = output_dir / "results_bundle.json"
        path.write_text(json.dumps(bundle))
        return bundle, path

    monkeypatch.setattr(lens_client, "run_lens", fake_lens)
    return seen


def run_stage(jobs, job_id):
    from orchestrator.src import worker

    code = worker.run_job(job_id)
    return code, jobs.get(job_id)


# ── store ────────────────────────────────────────────────────────────────────
def test_a_research_job_may_name_a_pathway_ask_for_one_or_turn_on_search(jobs):
    job = jobs.create("IOI", params={"pathway": "circuit_discovery"})
    assert (job.kind, job.stage, job.params_dict) == ("research", "scout",
                                                      {"pathway": "circuit_discovery"})
    with pytest.raises(ValueError, match="unknown pathway"):
        jobs.create("IOI", params={"pathway": "nope"})
    with pytest.raises(ValueError, match="params are only"):
        jobs.create("IOI", params={"pathway": "circuit_discovery", **IOI})
    assert jobs.create("IOI", params={"pathway": "auto", "search": True}).params_dict == {
        "pathway": "auto", "search": True}
    with pytest.raises(ValueError, match="search must be"):
        jobs.create("IOI", params={"search": "yes"})
    with pytest.raises(ValueError, match="only runs on gpt2"):
        jobs.create("IOI", model="pythia-70m", params={"pathway": "feature_explain"})


# ── Scout ────────────────────────────────────────────────────────────────────
def test_without_a_pathway_scout_is_unchanged(jobs, scout):
    agents = scout(plan(params=None))
    job = jobs.create("What heads mediate IOI?")

    code, after = run_stage(jobs, job.id)
    assert code == 0
    assert (after.stage, after.status) == ("lens", jobs.AWAITING)
    assert agents[0].messages == ["What heads mediate IOI?"]   # no brief, no revision
    assert after.params is None
    assert after.plan_path.endswith("ioi_research_plan.md")


def test_scout_chooses_the_pathways_inputs_and_they_are_saved(jobs, scout):
    agents = scout(plan(IOI))
    job = jobs.create("What heads mediate IOI?", params={"pathway": "circuit_discovery"})

    code, after = run_stage(jobs, job.id)
    assert code == 0
    assert (after.stage, after.status) == ("lens", jobs.AWAITING)
    assert after.params_dict == {"pathway": "circuit_discovery", "pathway_params": IOI}

    [message] = agents[0].messages
    assert message.startswith("What heads mediate IOI?")
    for needed in ("circuit_discovery", "corrupted_prompts", "leading space", "```json"):
        assert needed in message


def test_invalid_inputs_go_back_to_scout_until_it_fixes_them(jobs, scout):
    multi_token = {**IOI, "positive_tokens": [" Bartholomew"]}
    agents = scout(plan(multi_token), plan(IOI))
    job = jobs.create("IOI", params={"pathway": "circuit_discovery"})

    code, after = run_stage(jobs, job.id)
    assert code == 0
    assert after.params_dict["pathway_params"] == IOI
    first, revision = agents[0].messages
    assert "' Bartholomew' is 2 tokens" in revision
    assert "save_research_plan again" in revision


def test_inputs_still_invalid_after_two_revisions_fail_the_stage(jobs, scout):
    misaligned = {**IOI, "negative_tokens": [" John", " Anne"]}
    agents = scout(plan(None), plan({"prompts": ["p"]}), plan(misaligned))
    job = jobs.create("IOI", params={"pathway": "circuit_discovery"})

    code, after = run_stage(jobs, job.id)
    assert code == 1
    assert len(agents[0].messages) == 3                      # the ask, then two revisions
    assert "no ```json block" in agents[0].messages[1]
    assert "positive_tokens must be a non-empty list" in agents[0].messages[2]
    assert after.status == jobs.FAILED and after.stage == "scout"
    assert "still invalid after 2 revisions" in after.error
    assert "negative_tokens has 2 entries but prompts has 1" in after.error
    assert after.params_dict == {"pathway": "circuit_discovery"}   # nothing approved
    assert after.plan_path                                           # but the plan is visible


def test_a_block_for_the_wrong_pathway_is_refused():
    params, errors = catalog.extract_inputs("attribute_audit", plan(IOI), "gpt2")
    assert params is None
    assert "this job runs 'attribute_audit'" in errors[0]


def test_without_inputs_removes_only_the_pathway_block():
    other = '```json\n{"layers": [9, 10]}\n```'
    stripped = catalog.without_inputs(plan(IOI) + "\n" + other + "\n")
    assert IOI["prompts"][0] not in stripped
    assert other in stripped
    assert "## Experiments" in stripped


# ── Lens and Quill ───────────────────────────────────────────────────────────
def scouted(jobs, scout):
    scout(plan(IOI))
    job = jobs.create("IOI", params={"pathway": "circuit_discovery"})
    run_stage(jobs, job.id)
    jobs.approve(job.id)
    return job.id


def test_lens_runs_the_pathway_first_on_the_approved_inputs(jobs, scout, pathway_runs, lens):
    job_id = scouted(jobs, scout)

    code, after = run_stage(jobs, job_id)
    assert code == 0
    assert (after.stage, after.status) == ("quill", jobs.AWAITING)
    assert pathway_runs == [IOI]
    assert "Pathway already run" in lens["plans"][0] and "circuit_found" in lens["plans"][0]
    # The inputs are gone from Lens's copy, so it can't rerun them as an extra.
    assert "```json" not in lens["plans"][0]
    assert IOI["prompts"][0] not in lens["plans"][0]
    assert "Pathway inputs removed" in lens["plans"][0]

    saved = json.loads((jobs.job_dir(job_id) / "pathway" / "result.json").read_text())
    assert (saved["pathway"], saved["version"], saved["params"]) == (
        "circuit_discovery", catalog.version("circuit_discovery"), IOI)

    bundle = json.loads(open(after.bundle_path).read())
    first, extra = bundle["results"]
    assert (first["role"], first["name"], first["tool"]) == (
        "pathway", "pathway:circuit_discovery", f"circuit_discovery@v{catalog.version('circuit_discovery')}")
    assert first["data"]["verdict"] == "circuit_found"
    assert first["tool_kwargs"] == IOI
    assert (extra["role"], extra["name"]) == ("extra", "logit_lens_ioi")
    assert (bundle["n_total"], bundle["n_success"]) == (2, 2)
    assert bundle["pathway"]["verdict"] == "circuit_found"

    from quill.mcp_server.src.app.helpers import bundle_to_context
    from quill.mcp_server.src.models.schemas import ExperimentBundle

    context = bundle_to_context(ExperimentBundle.from_json(after.bundle_path))
    assert "primary evidence" in context
    assert context.index("Pathway result (primary evidence): pathway:circuit_discovery") \
        < context.index("Extra experiment: logit_lens_ioi")


def test_a_retried_lens_stage_reuses_the_pathway_result(jobs, scout, pathway_runs, lens):
    job_id = scouted(jobs, scout)
    lens["fail"] = True
    code, after = run_stage(jobs, job_id)
    assert code == 1 and after.stage == "lens"

    lens["fail"] = False
    jobs.retry_if_failed(job_id)
    code, after = run_stage(jobs, job_id)
    assert code == 0
    assert pathway_runs == [IOI]                     # ran once, across both attempts
    assert json.loads(open(after.bundle_path).read())["results"][0]["role"] == "pathway"


def test_changed_inputs_rerun_the_pathway(jobs, scout, pathway_runs, lens):
    job_id = scouted(jobs, scout)
    lens["fail"] = True
    run_stage(jobs, job_id)

    # Inputs edited at the gate: the saved result no longer matches them.
    edited = {**IOI, "prompts": ["When Anne and Tom met, Tom thanked"]}
    params = {**jobs.get(job_id).params_dict, "pathway_params": edited}
    jobs.update(job_id, params=json.dumps(params))
    lens["fail"] = False
    jobs.retry_if_failed(job_id)
    run_stage(jobs, job_id)
    assert pathway_runs == [IOI, edited]


# ── Reruns ───────────────────────────────────────────────────────────────────
def test_a_rerun_runs_the_pathway_again_on_the_same_inputs(jobs, scout, pathway_runs, lens):
    job_id = scouted(jobs, scout)
    run_stage(jobs, job_id)
    source = jobs.get(job_id)

    again = jobs.rerun(job_id)
    assert (again.stage, again.status, again.model) == ("lens", jobs.QUEUED, "gpt2")
    assert again.plan_path == source.plan_path
    assert again.params_dict == {"pathway": "circuit_discovery", "pathway_params": IOI,
                                 "rerun_of": job_id}

    code, after = run_stage(jobs, again.id)
    assert code == 0
    assert pathway_runs == [IOI, IOI]                # its own run, not the source's result
    assert len(lens["plans"]) == 2
    saved = json.loads((jobs.job_dir(again.id) / "pathway" / "result.json").read_text())
    assert (saved["verdict"], saved["params"]) == ("circuit_found", IOI)


def test_a_rerun_can_move_the_inputs_to_another_model(jobs, scout, pathway_runs, lens):
    job_id = scouted(jobs, scout)
    again = jobs.rerun(job_id, model="pythia-70m")
    assert again.model == "pythia-70m"

    code, after = run_stage(jobs, again.id)
    assert code == 0
    saved = json.loads((jobs.job_dir(again.id) / "pathway" / "result.json").read_text())
    assert saved["model"] == "pythia-70m" and saved["params"] == IOI
    assert "**Target Model (required, overrides any other model mentioned above)**: pythia-70m" \
        in lens["plans"][-1]


def test_inputs_that_dont_fit_the_new_model_are_refused(jobs, scout, monkeypatch):
    job_id = scouted(jobs, scout)

    class Splits:
        """A tokenizer for which every answer is two tokens."""

        def encode(self, text, add_special_tokens=False):
            return [0, 1]

        def convert_ids_to_tokens(self, ids):
            return ["a", "b"]

    monkeypatch.setattr(catalog, "_tokenizer",
                        lambda model: Splits() if model == "pythia-70m" else FakeTokenizer())
    with pytest.raises(ValueError, match="can't run on pythia-70m.*' Mary' is 2 tokens"):
        jobs.rerun(job_id, model="pythia-70m")

    params = {**jobs.get(job_id).params_dict, "pathway": "feature_explain"}
    jobs.update(job_id, params=json.dumps(params))
    with pytest.raises(ValueError, match="feature_explain only runs on gpt2"):
        jobs.rerun(job_id, model="gpt2-medium")


def test_a_rerun_rechecks_inputs_that_reach_lens_on_another_model(jobs, scout, pathway_runs,
                                                                  lens, monkeypatch):
    # The dashboard queues a rerun without the tokenizer; Lens checks it.
    job_id = scouted(jobs, scout)
    again = jobs.rerun(job_id)
    jobs.update(again.id, model="gpt2-medium",
                params=json.dumps({**again.params_dict, "pathway": "feature_explain"}))

    code, after = run_stage(jobs, again.id)
    assert code == 1
    assert "the approved feature_explain inputs can't run on gpt2-medium" in after.error
    assert pathway_runs == []


def test_only_a_job_with_approved_inputs_can_be_rerun(jobs, scout):
    with pytest.raises(ValueError, match="no job"):
        jobs.rerun("nope")
    plain = jobs.create("What heads mediate IOI?")
    with pytest.raises(ValueError, match="no approved pathway inputs"):
        jobs.rerun(plain.id)
    not_scouted = jobs.create("IOI", params={"pathway": "circuit_discovery"})
    with pytest.raises(ValueError, match="no approved pathway inputs"):
        jobs.rerun(not_scouted.id)


def test_a_rerun_replays_the_source_jobs_experiments(jobs, scout, pathway_runs, lens):
    job_id = scouted(jobs, scout)
    run_stage(jobs, job_id)
    assert lens["replays"] == [None]                 # the source read its plan

    again = jobs.rerun(job_id, model="pythia-70m")
    code, after = run_stage(jobs, again.id)
    assert code == 0
    replay = lens["replays"][-1]
    assert (replay["research_question"], replay["model_name"]) == ("IOI", "pythia-70m")
    # The extra, as it ran, on the rerun's model; the pathway entry isn't replayed.
    assert replay["experiments"] == [{"name": "logit_lens_ioi", "tool": "logit_lens",
                                      "model_name": "pythia-70m", "prompts": IOI["prompts"],
                                      "tool_kwargs": {}}]
    bundle = json.loads(open(after.bundle_path).read())
    assert bundle["replayed_from"] == job_id
    assert [r["role"] for r in bundle["results"]] == ["pathway", "extra"]


def test_a_rerun_reads_the_plan_again_when_the_source_bundle_is_gone(jobs, scout,
                                                                     pathway_runs, lens):
    from shared.db import events

    job_id = scouted(jobs, scout)
    run_stage(jobs, job_id)
    Path(jobs.get(job_id).bundle_path).unlink()

    again = jobs.rerun(job_id)
    code, after = run_stage(jobs, again.id)
    assert code == 0
    assert lens["replays"][-1] is None
    note = next(e for e in events.list_events(again.id) if e["kind"] == "note")
    assert note["status"] == "error" and "reads the plan again" in note["summary"]



# ── Scout chooses the pathway ────────────────────────────────────────────────
def choice(name, params=None, why="it asks which heads carry IOI"):
    """A Scout plan under "auto": a pathway (or null) with a reason and inputs."""
    block = {"pathway": name, "why": why, **({"params": params} if params is not None else {})}
    return plan(params=None) + "\n```json\n" + json.dumps(block) + "\n```\n"


def test_under_auto_scout_picks_the_pathway_and_its_inputs(jobs, scout):
    agents = scout(choice("circuit_discovery", IOI))
    job = jobs.create("What heads mediate IOI?", params={"pathway": "auto"})

    code, after = run_stage(jobs, job.id)
    assert code == 0 and (after.stage, after.status) == ("lens", jobs.AWAITING)
    assert "## Choose a pathway (or none)" in agents[0].messages[0]
    assert "attribute_audit" not in agents[0].messages[0]          # the old name isn't offered
    assert after.params_dict == {"pathway": "circuit_discovery", "pathway_params": IOI,
                                 "pathway_request": "auto", "pathway_why": "it asks which heads carry IOI"}


def test_under_auto_scout_may_choose_no_pathway(jobs, scout, lens):
    scout(choice(None, why="it's about training data, which no pathway looks at"))
    job = jobs.create("Which training documents taught GPT-2 IOI?", params={"pathway": "auto"})

    code, after = run_stage(jobs, job.id)
    assert code == 0
    assert after.params_dict == {"pathway_request": "auto",
                                 "pathway_why": "it's about training data, which no pathway looks at"}
    # Lens gets the plan without the choice block, and no pathway runs.
    jobs.approve(job.id)
    code, after = run_stage(jobs, job.id)
    assert code == 0
    assert '"pathway": null' not in lens["plans"][0] and "## Experiments" in lens["plans"][0]


def test_an_unusable_choice_is_sent_back_for_revision(jobs, scout):
    agents = scout(choice("nope", IOI), choice("circuit_discovery", IOI, why=""),
                   choice("circuit_discovery", IOI))
    job = jobs.create("IOI", params={"pathway": "auto"})

    code, after = run_stage(jobs, job.id)
    assert code == 0
    assert "unknown pathway 'nope'" in agents[0].messages[1]
    assert 'needs a "why"' in agents[0].messages[2]
    assert after.params_dict["pathway"] == "circuit_discovery"


def test_a_retried_auto_scout_chooses_again(jobs, scout):
    scout(choice("circuit_discovery", IOI))
    job = jobs.create("IOI", params={"pathway": "auto"})
    run_stage(jobs, job.id)

    from orchestrator.src import worker

    agents = scout(choice(None, why="on reflection, nothing fits"))
    worker.run_scout_stage(jobs.get(job.id))                       # what a retried stage runs
    after = jobs.get(job.id)
    assert "## Choose a pathway" in agents[-1].messages[0]         # asked to choose, not bound
    assert after.params_dict == {"pathway_request": "auto", "pathway_why": "on reflection, nothing fits"}


def test_search_is_off_unless_the_job_asks_for_it(jobs, scout):
    agents = scout(plan(params=None))
    run_stage(jobs, jobs.create("IOI").id)
    scout(plan(params=None))                                # same list: agents[1] is the next job's
    run_stage(jobs, jobs.create("IOI", params={"search": True}).id)
    assert agents[0].search is False and agents[1].search is True


def test_the_choice_brief_lists_each_pathway_with_its_inputs():
    brief = catalog.choice_brief("gpt2")
    for name in ("circuit_discovery", "concept_direction", "model_diffing", "feature_explain"):
        assert f"`{name}`" in brief
    assert "`other_model` (one string)" in brief
    assert "`feature_explain`" not in catalog.choice_brief("pythia-70m")   # its SAE is gpt2's
