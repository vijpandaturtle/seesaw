"""A job's timeline: one row per thing its worker did, written as it happens.

The worker log is whatever got printed. This is the structured record next to
it: each stage, each LangGraph node, each LLM call (model, tokens, what it
answered), each tool call (arguments, result), each Lens experiment, each
pathway step, and every check that failed — with when it started, how long
it took and how it ended. The dashboard reads it to show a live timeline, so
anyone can trace how a result was produced, step by step.

    from shared.db import events

    with events.bind(job.id, "lens"):
        with events.span("experiment", "ioi_ablation") as s:
            ...
            s.summary = "success, 2 plots"
        events.emit("check", "pathway inputs", status="error", data={"errors": errs})

    events.list_events(job.id)      # oldest first

Nothing is recorded unless a job is bound, so the agents run unchanged outside
a worker (tests, notebooks, the MCP servers). Writing an event never raises: a
timeline that can't be written costs a warning, not the job.

Events nest. Each one's parent is the span open when it started (the stage
around a node, the node around an LLM call), so the dashboard can indent them.
One worker runs one job at a time, so the open spans are kept per process
rather than per thread: LangGraph may run a node on another thread, and its
events still belong under that node.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from dataclasses import dataclass, field

from . import jobs

# Event kinds. A free-form string in the table; these are the ones written.
STAGE      = "stage"         # one job stage: scout, lens, quill, pathway
NODE       = "node"          # one LangGraph node
LLM        = "llm"           # one chat-model call
TOOL       = "tool"          # one agent tool call (Scout's searches, its save)
EXPERIMENT = "experiment"    # one Lens experiment
PATHWAY    = "pathway"       # one pathway run, around its steps
STEP       = "step"          # one pathway step
CHECK      = "check"         # a validation: Scout's pathway inputs
NOTE       = "note"          # a decision worth seeing: what a rerun replays
STATUS     = "status"        # the job changed state: awaiting approval, done
ERROR      = "error"

# Event statuses.
RUNNING = "running"          # a span that hasn't finished (or whose worker died)
OK      = "ok"
FAILED  = "error"

# Long strings are cut so one LLM answer can't bloat the table; the artifacts
# on disk keep everything.
MAX_TEXT  = 4_000
MAX_ITEMS = 50
MAX_DATA  = 32_000
MAX_SUMMARY = 500

_lock = threading.Lock()
_bound: dict | None = None      # {"job_id", "stage"} while a worker runs a stage
_open: list[int] = []           # ids of the spans open right now, innermost last
_warned = False


# ── Binding ──────────────────────────────────────────────────────────────────
@contextlib.contextmanager
def bind(job_id: str, stage: str | None = None):
    """Record events for this job (and stage) until the block exits."""
    global _bound
    previous = _bound
    _bound = {"job_id": job_id, "stage": stage}
    try:
        yield
    finally:
        _bound = previous


def bound_job() -> str | None:
    return _bound["job_id"] if _bound else None


# ── Writing ──────────────────────────────────────────────────────────────────
def emit(kind: str, name: str, *, status: str = OK, summary: str | None = None,
         data: dict | None = None, duration_ms: float | None = None,
         parent_id: int | None = None) -> int | None:
    """Record one finished event. Returns its id, or None when nothing is bound."""
    return _insert(kind, name, status, summary, data, duration_ms, parent_id)


def start(kind: str, name: str, *, summary: str | None = None, data: dict | None = None,
          parent_id: int | None = None, nest: bool = True) -> int | None:
    """Record a span that has started; finish() it later.

    With `nest`, events recorded until it finishes go under it. The LangChain
    callbacks pass nest=False for LLM and tool calls, which can run in
    parallel, and parent those by run id instead.
    """
    event_id = _insert(kind, name, RUNNING, summary, data, None, parent_id)
    if event_id is not None and nest:
        with _lock:
            _open.append(event_id)
    return event_id


def finish(event_id: int | None, status: str = OK, *, duration_ms: float | None = None,
           summary: str | None = None, data: dict | None = None) -> None:
    """Close a span: its status, how long it took, and what it found."""
    if event_id is None:
        return
    with _lock:
        if event_id in _open:
            _open.remove(event_id)
    fields: dict = {"status": status}
    if duration_ms is not None:
        fields["duration_ms"] = round(duration_ms, 1)
    if summary is not None:
        fields["summary"] = _clip_text(summary, MAX_SUMMARY)
    if data is not None:
        fields["data"] = _encode(data)
    assignments = ", ".join(f"{k} = ?" for k in fields)
    _write(f"UPDATE job_events SET {assignments} WHERE id = ?", (*fields.values(), event_id))


@dataclass
class Span:
    """What a span() block can fill in before it closes."""
    id: int | None
    summary: str | None = None
    data: dict = field(default_factory=dict)
    status: str = OK


@contextlib.contextmanager
def span(kind: str, name: str, *, summary: str | None = None, data: dict | None = None):
    """A span around a block: started on entry, finished on exit.

    An exception marks it failed, with the error as its summary, and is
    re-raised.
    """
    s = Span(start(kind, name, summary=summary, data=data), data=dict(data or {}))
    t0 = time.perf_counter()
    try:
        yield s
    except BaseException as exc:
        finish(s.id, FAILED, duration_ms=_ms(t0),
               summary=f"{type(exc).__name__}: {exc}", data=s.data or None)
        raise
    finish(s.id, s.status, duration_ms=_ms(t0), summary=s.summary, data=s.data or None)


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000


def _insert(kind, name, status, summary, data, duration_ms, parent_id) -> int | None:
    bound = _bound
    if bound is None:
        return None
    if parent_id is None:
        with _lock:
            parent_id = _open[-1] if _open else None
    row = (bound["job_id"], parent_id, time.time(), bound["stage"], kind, _clip_text(name, 200),
           status, None if duration_ms is None else round(duration_ms, 1),
           None if summary is None else _clip_text(summary, MAX_SUMMARY),
           None if data is None else _encode(data))
    sql = ("INSERT INTO job_events (job_id, parent_id, ts, stage, kind, name, status,"
           " duration_ms, summary, data) VALUES (?,?,?,?,?,?,?,?,?,?)")
    try:
        with jobs.connect() as conn:
            if jobs.is_postgres():
                return jobs._ex(conn, sql + " RETURNING id", row).fetchone()["id"]
            return jobs._ex(conn, sql, row).lastrowid
    except Exception as exc:                          # noqa: BLE001 — never fail the job
        _warn(exc)
        return None


def _write(sql: str, params: tuple) -> None:
    try:
        with jobs.connect() as conn:
            jobs._ex(conn, sql, params)
    except Exception as exc:                          # noqa: BLE001 — never fail the job
        _warn(exc)


def _warn(exc: Exception) -> None:
    global _warned
    if not _warned:
        _warned = True
        print(f"⚠️  couldn't write to the job timeline ({type(exc).__name__}: {exc}); "
              "the job carries on without it")


# ── Encoding ─────────────────────────────────────────────────────────────────
def _clip_text(text: str, limit: int = MAX_TEXT) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + f"… [{len(text) - limit} more chars]"


def clip(value, depth: int = 0):
    """A JSON-safe copy of value with long strings and lists cut short."""
    if depth > 6:
        return _clip_text(repr(value), 200)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _clip_text(value)
    if isinstance(value, dict):
        items = list(value.items())
        out = {str(k): clip(v, depth + 1) for k, v in items[:MAX_ITEMS]}
        if len(items) > MAX_ITEMS:
            out["…"] = f"{len(items) - MAX_ITEMS} more keys"
        return out
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        out = [clip(v, depth + 1) for v in items[:MAX_ITEMS]]
        if len(items) > MAX_ITEMS:
            out.append(f"… {len(items) - MAX_ITEMS} more")
        return out
    if hasattr(value, "tolist"):                      # numpy arrays and scalars, tensors
        return clip(value.tolist(), depth + 1)
    return _clip_text(str(value))


def _encode(data: dict) -> str:
    text = json.dumps(clip(data))
    if len(text) > MAX_DATA:
        text = json.dumps({"truncated": True, "preview": text[:MAX_DATA]})
    return text


# ── Reading ──────────────────────────────────────────────────────────────────
def list_events(job_id: str, with_data: bool = True) -> list[dict]:
    """A job's events, oldest first, with `data` decoded."""
    columns = "*" if with_data else (
        "id, job_id, parent_id, ts, stage, kind, name, status, duration_ms, summary")
    with jobs.connect() as conn:
        rows = jobs._ex(conn, f"SELECT {columns} FROM job_events WHERE job_id = ? ORDER BY id",
                        (job_id,)).fetchall()
    out = []
    for row in rows:
        event = {k: row[k] for k in row.keys()}
        if event.get("data"):
            event["data"] = json.loads(event["data"])
        out.append(event)
    return out


# ── LangChain ────────────────────────────────────────────────────────────────
def callbacks() -> list:
    """Callbacks that record a LangGraph run's nodes, LLM calls and tool calls.

    Pass as config["callbacks"]. Empty when no job is bound, so a client that
    always passes them changes nothing outside a worker.
    """
    if _bound is None:
        return []
    return [_handler_class()()]


_HANDLER = None


def _handler_class():
    """Built on first use so importing this module doesn't import LangChain."""
    global _HANDLER
    if _HANDLER is not None:
        return _HANDLER

    from langchain_core.callbacks import BaseCallbackHandler

    class TimelineHandler(BaseCallbackHandler):
        """Turns LangChain's run callbacks into timeline spans.

        Each run is matched to its span by run id, and nested under its parent
        run's span when that one is on the timeline too (an LLM call inside a
        node), else under whatever span is open (the stage).
        """

        def __init__(self):
            # run_id -> (event id, start time, data it started with). A run
            # with no start time borrows its parent's span and isn't finished.
            self._spans: dict = {}

        def _start(self, run_id, parent_run_id, kind, name, data=None):
            parent = self._spans.get(parent_run_id, (None,))[0] if parent_run_id else None
            # A node nests what the code inside it records (a Lens experiment);
            # LLM and tool calls can run in parallel, so they don't.
            event_id = start(kind, name, data=data, parent_id=parent, nest=kind == NODE)
            self._spans[run_id] = (event_id, time.perf_counter(), data or {})

        def _finish(self, run_id, status=OK, summary=None, data=None):
            event_id, t0, started = self._spans.pop(run_id, (None, None, None))
            if t0 is not None:
                finish(event_id, status, duration_ms=_ms(t0), summary=summary,
                       data={**started, **data} if data else None)

        # Graph nodes. LangGraph reports each node as a chain named after it;
        # everything else it reports as a chain (the graph, channel writes) is
        # plumbing and left off the timeline.
        def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None,
                           metadata=None, **kwargs):
            node = (metadata or {}).get("langgraph_node")
            name = kwargs.get("name") or (serialized or {}).get("name")
            if node and name == node and not node.startswith("__"):
                self._start(run_id, parent_run_id, NODE, node)
            elif parent_run_id in self._spans:
                # A chain inside a node: keep its LLM calls under the node.
                self._spans[run_id] = (self._spans[parent_run_id][0], None, None)

        def on_chain_end(self, outputs, *, run_id, **kwargs):
            self._finish(run_id)

        def on_chain_error(self, error, *, run_id, **kwargs):
            self._finish(run_id, FAILED, summary=f"{type(error).__name__}: {error}")

        # Model calls.
        def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None,
                                metadata=None, invocation_params=None, **kwargs):
            params = invocation_params or kwargs.get("invocation_params") or {}
            model = (params.get("model") or params.get("model_name")
                     or (metadata or {}).get("ls_model_name") or "llm")
            conversation = messages[0] if messages else []
            self._start(run_id, parent_run_id, LLM, model, data={
                "model": model,
                "messages": len(conversation),
                "prompt": _message_text(conversation[-1]) if conversation else "",
            })

        def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs):
            self._start(run_id, parent_run_id, LLM, (serialized or {}).get("name", "llm"),
                        data={"prompt": prompts[-1] if prompts else ""})

        def on_llm_end(self, response, *, run_id, **kwargs):
            generation = response.generations[0][0] if response.generations else None
            message = getattr(generation, "message", None)
            usage = getattr(message, "usage_metadata", None) or {}
            tool_calls = [{"name": c.get("name"), "args": c.get("args")}
                          for c in getattr(message, "tool_calls", None) or []]
            text = _message_text(message) if message is not None else getattr(generation, "text", "")
            parts = []
            if usage:
                parts.append(f"{usage.get('input_tokens', 0):,} in / "
                             f"{usage.get('output_tokens', 0):,} out tokens")
            if tool_calls:
                parts.append("calls " + ", ".join(c["name"] or "?" for c in tool_calls))
            elif text:
                parts.append(text.strip().replace("\n", " ")[:160])
            self._finish(run_id, summary=" · ".join(parts) or None,
                         data={"response": text, "tool_calls": tool_calls, "usage": dict(usage)})

        def on_llm_error(self, error, *, run_id, **kwargs):
            self._finish(run_id, FAILED, summary=f"{type(error).__name__}: {error}")

        # Agent tools.
        def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None,
                          inputs=None, **kwargs):
            name = (serialized or {}).get("name") or kwargs.get("name") or "tool"
            self._start(run_id, parent_run_id, TOOL, name,
                        data={"input": inputs if inputs is not None else input_str})

        def on_tool_end(self, output, *, run_id, **kwargs):
            text = _message_text(output) if hasattr(output, "content") else str(output)
            self._finish(run_id, summary=text.strip().replace("\n", " ")[:160] or None,
                         data={"output": text})

        def on_tool_error(self, error, *, run_id, **kwargs):
            self._finish(run_id, FAILED, summary=f"{type(error).__name__}: {error}")

    _HANDLER = TimelineHandler
    return _HANDLER


def _message_text(message) -> str:
    """A message's text, whether its content is a string or a list of blocks."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, dict) and block.get("type") == "tool_use":
                parts.append(f"[tool_use {block.get('name')}]")
        return "\n".join(parts)
    return str(content)
