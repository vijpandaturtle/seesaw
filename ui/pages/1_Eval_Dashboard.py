"""Eval dashboard — fixture health, task-run scores, and per-run drilldown.

Appears as a page of the main Seesaw app:
    streamlit run ui/app.py       (sidebar → "Eval Dashboard")

Reads persisted task runs from outputs/eval_runs/*.json (written by
`python -m eval.runner <task_id>`) and runs the offline fixture check live
(it's fast and free — no LLM calls).
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.fixtures import check_offline, load_all as load_fixtures  # noqa: E402
from eval.tasks import load_all as load_tasks                       # noqa: E402

RUNS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "eval_runs"

st.set_page_config(page_title="Seesaw · Eval", page_icon="📊", layout="wide")
st.title("📊 Eval Dashboard")

tab_runs, tab_matrix, tab_fixtures, tab_tasks = st.tabs(
    ["Task runs", "Score matrix", "Fixture health", "Task catalog"]
)


# ── helpers ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_runs() -> list[dict]:
    runs = []
    for p in sorted(RUNS_DIR.glob("*.json")):
        try:
            runs.append({"_file": p.name, **json.loads(p.read_text())})
        except (json.JSONDecodeError, OSError):
            continue
    return runs


def score_icon(score) -> str:
    if score is None:
        return "⚠️"
    return "✅" if score >= 0.5 else "❌"


# ── Tab 1: task runs ─────────────────────────────────────────────────────────
with tab_runs:
    runs = load_runs()
    if not runs:
        st.info("No runs yet — execute one with:  `python -m eval.runner <task_id>`")
    else:
        rows = []
        for r in runs:
            scores = [s for s in r.get("scores", {}).values() if s is not None]
            rows.append({
                "run": r["_file"],
                "task": r.get("task_id"),
                "target": r.get("target"),
                "graders": len(r.get("grader_results", [])),
                "mean score": round(sum(scores) / len(scores), 2) if scores else None,
                "when": time.strftime("%Y-%m-%d %H:%M", time.localtime(r.get("started_at", 0))),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        pick = st.selectbox("Inspect a run", [r["_file"] for r in runs][::-1])
        run = next(r for r in runs if r["_file"] == pick)

        st.subheader(f"{run['task_id']} — grader results")
        for g in run.get("grader_results", []):
            with st.expander(f"{score_icon(g.get('score'))} {g['criterion']}  ·  score={g.get('score')}"):
                if g.get("detail"):
                    st.write(g["detail"])
                if g.get("error"):
                    st.error(g["error"])
                if g.get("metrics"):
                    st.json(g["metrics"], expanded=False)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Tracked metrics")
            st.json(run.get("metrics", {}))
        with col2:
            st.subheader("Trajectory")
            traj = run.get("trajectory", [])
            if traj:
                st.dataframe(pd.DataFrame([{"tool": t.get("name")} for t in traj]),
                             use_container_width=True, hide_index=True)
            else:
                st.caption("no tool calls recorded")
        if run.get("errors"):
            st.error("Run errors: " + "; ".join(run["errors"]))


# ── Tab 2: score matrix across runs ──────────────────────────────────────────
with tab_matrix:
    runs = load_runs()
    if not runs:
        st.info("No runs to compare yet.")
    else:
        cells = {}
        for r in runs:
            for crit, score in r.get("scores", {}).items():
                cells.setdefault(crit, {})[r["task_id"]] = score
        df = pd.DataFrame(cells).T.sort_index()
        st.caption("criterion × task — latest persisted runs (None = needs human / not applicable)")
        st.dataframe(
            df.style.map(
                lambda v: "background-color:#1a4d1a" if isinstance(v, (int, float)) and v >= 0.5
                else ("background-color:#5c1a1a" if isinstance(v, (int, float)) else "")
            ),
            use_container_width=True,
        )


# ── Tab 3: fixture health (grader meta-eval) ─────────────────────────────────
with tab_fixtures:
    st.caption("Offline fixtures run live on page load (free). LLM fixtures show last-known status; "
               "re-verify with `check_llm()` — those cost judge calls.")
    if st.button("Run offline fixture check"):
        st.session_state["fixture_result"] = check_offline()
    res = st.session_state.get("fixture_result")
    if res:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total fixtures", res["total"])
        c2.metric("Ran offline", res["ran_offline"])
        c3.metric("Mismatches", len(res["mismatches"]))
        if res["mismatches"]:
            st.error("Mismatches — a grader (or fixture label) is broken:")
            st.json(res["mismatches"])
        cov_rows = [{"grader": g, "labels": ", ".join(sorted(ls)),
                     "balanced": "✅" if {"pass", "fail"} <= set(ls) else ("👤 human" if ls == ["na"] else "❌")}
                    for g, ls in sorted(res["coverage"].items())]
        st.dataframe(pd.DataFrame(cov_rows), use_container_width=True, hide_index=True)
    else:
        fx = load_fixtures()
        st.metric("Fixtures on disk", len(fx))


# ── Tab 4: task catalog ──────────────────────────────────────────────────────
with tab_tasks:
    tasks = load_tasks()
    run_ids = {r.get("task_id") for r in load_runs()}
    rows = [{
        "task": t.id,
        "target": t.target or "-",
        "graders": ", ".join(g.type for g in t.graders),
        "ran?": "✅" if t.id in run_ids else "—",
    } for t in tasks.values()]
    st.dataframe(pd.DataFrame(rows).sort_values("task"), use_container_width=True, hide_index=True)
    st.caption("Run any task with:  `python -m eval.runner <task_id>`")
