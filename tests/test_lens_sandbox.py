"""A timed-out experiment can't leak into the next one.

The failure this guards: a stuck image_token_ablation kept running on the
shared model after the sandbox gave up on it, and its hooks broke every later
experiment ("mask [1, 749] … tensor [1, 745, …]"). After a timeout the model
is retired: the stuck thread stops at its next forward call, its hooks come
off, and the next experiment gets a fresh copy.
"""

from __future__ import annotations

import threading
import time

import pytest

torch = pytest.importorskip("torch")
session = pytest.importorskip("lens.mcp_server.src.app.model_session")
from lens.mcp_server.src.app.sandbox import run_in_sandbox  # noqa: E402


class Handle:
    """Stands in for a lever handle: the nn.Module lives on .model."""

    def __init__(self):
        self.model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())


def test_retiring_stops_the_stuck_thread_and_removes_its_hooks(monkeypatch):
    handle = Handle()
    monkeypatch.setitem(session._MODEL_CACHE, ("lever", "tiny"), handle)
    stopped = threading.Event()
    seen: dict = {}

    def stuck_tool(model, prompts):
        """Holds a hook on the model and keeps running forwards, like a layer sweep."""
        hook = model.model[0].register_forward_hook(lambda m, i, o: o * 0)
        try:
            while True:
                model.model(torch.ones(1, 4))
                time.sleep(0.01)
        except session.ModelRetired as e:
            seen["error"] = e
        finally:
            hook.remove()
            stopped.set()

    result = run_in_sandbox(stuck_tool, {"model": handle, "prompts": ["p"]}, "stuck", timeout=0.2)
    assert result.status == "timeout"
    assert not stopped.is_set()          # still running on the shared model

    session.retire_model("tiny", handle)
    assert stopped.wait(5), "the stuck thread should fail at its next forward call"
    assert isinstance(seen["error"], session.ModelRetired)
    assert ("lever", "tiny") not in session._MODEL_CACHE
    assert not handle.model[0]._forward_hooks    # the tool's own hook came off on the way out


def test_retiring_a_bare_module_works_too(monkeypatch):
    model = torch.nn.Linear(2, 2)
    monkeypatch.setitem(session._MODEL_CACHE, ("seesaw", "gpt2"), model)
    session.retire_model("gpt2", model)
    assert ("seesaw", "gpt2") not in session._MODEL_CACHE
    with pytest.raises(session.ModelRetired):
        model(torch.ones(1, 2))


def test_retiring_leaves_a_different_cached_copy_alone(monkeypatch):
    old, new = Handle(), Handle()
    monkeypatch.setitem(session._MODEL_CACHE, ("lever", "tiny"), new)
    session.retire_model("tiny", old)    # already replaced: don't evict the replacement
    assert session._MODEL_CACHE[("lever", "tiny")] is new


def test_run_experiment_retires_on_timeout_and_records_kwargs(monkeypatch):
    import importlib

    node = importlib.import_module("lens.mcp_server.src.nodes.run_experiment")   # the package re-exports the function under this name

    handle = Handle()
    monkeypatch.setenv("SEESAW_INTERP_BACKEND", "lever")
    monkeypatch.setattr(node, "get_model", lambda name: handle)
    monkeypatch.setattr(node, "SWEEP_TIMEOUT", 0.2)
    retired = []
    monkeypatch.setattr(node, "retire_model", lambda name, m: retired.append((name, m)))
    release = threading.Event()

    def slow(model, prompts, images, positive_tokens, negative_tokens, output_dir=None):
        release.wait(5)

    monkeypatch.setattr(node, "registry", lambda: {"image_token_ablation": slow})
    monkeypatch.setattr(node, "normalize_tool_kwargs",
                        lambda tool, kw, n: (dict(kw), [], []))
    spec = {"name": "sweep", "tool": "image_token_ablation", "model_name": "tiny",
            "prompts": ["what colour?"],
            "tool_kwargs": {"images": ["solid:red"], "positive_tokens": ["red"], "negative_tokens": ["blue"]}}
    out = node.run_experiment({"experiment_queue": [spec]})
    release.set()
    assert out["last_result"]["status"] == "timeout"
    assert retired == [("tiny", handle)]
    assert out["last_result"]["tool_kwargs"]["positive_tokens"] == ["red"]


def test_quill_reads_a_bundle_with_tool_kwargs_and_unknown_fields(tmp_path):
    import json

    from quill.mcp_server.src.app.helpers import format_result_for_context
    from quill.mcp_server.src.models.schemas import ExperimentBundle

    path = tmp_path / "bundle.json"
    path.write_text(json.dumps({"research_question": "q", "model_name": "m", "results": [
        {"name": "e", "tool": "logit_lens", "model_name": "m", "prompts": ["p"],
         "tool_kwargs": {"positive_tokens": ["red"]}, "added_later": 1}]}))
    bundle = ExperimentBundle.from_json(path)
    assert "positive_tokens" in format_result_for_context(bundle.results[0])
