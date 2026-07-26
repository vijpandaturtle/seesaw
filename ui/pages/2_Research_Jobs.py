"""Jobs dashboard — submit research runs, watch them, approve each stage.

Appears as a page of the main Seesaw app:
    streamlit run ui/app.py       (sidebar → "Research Jobs")

Unlike the single-run flow on the main page, stages here execute in a
detached worker process (orchestrator/src/worker.py) and all state lives
in SQLite, so a run keeps going after you close the tab. The HITL gates
become approve buttons on a job that is parked in awaiting_approval.
"""

import json
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.db import jobs  # noqa: E402

MODEL_OPTIONS = [
    "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
    "pythia-70m", "pythia-160m", "pythia-410m", "gpt-neo-125m",
]

STATUS_ICON = {
    jobs.QUEUED:    "⏳",
    jobs.RUNNING:   "🔄",
    jobs.AWAITING:  "⏸️",
    jobs.DONE:      "✅",
    jobs.FAILED:    "❌",
    jobs.CANCELLED: "🚫",
}

# What the human is being asked to approve at each gate.
GATE_PROMPT = {
    "lens":  "Approve Scout's research plan and run Lens experiments?",
    "quill": "Send these experiment results to Quill for critique?",
}

st.set_page_config(page_title="Seesaw · Jobs", page_icon="🗂️", layout="wide")
st.title("🗂️ Research Jobs")


# ── New job ──────────────────────────────────────────────────────────────────
with st.expander("➕ New research job", expanded=not jobs.list_jobs(1)):
    with st.form("new_job", clear_on_submit=True):
        question = st.text_area(
            "Research question",
            placeholder="What attention heads mediate indirect object identification in GPT-2 Small?",
            height=80,
        )
        c1, c2 = st.columns([2, 3])
        model = c1.selectbox("Target model", MODEL_OPTIONS, help="Passed to Lens as an explicit override.")
        auto = c2.checkbox(
            "Run all stages without stopping",
            help="Skip the approval gates between Scout, Lens, and Quill.",
        )
        if st.form_submit_button("Start job", type="primary") and question.strip():
            job = jobs.create(question, model=model, auto_approve=auto)
            jobs.spawn_worker(job.id)
            st.success(f"Job `{job.id}` started.")
            time.sleep(0.5)     # let the worker write its first log line
            st.rerun()


# ── Overview ─────────────────────────────────────────────────────────────────
all_jobs = jobs.list_jobs()
if not all_jobs:
    st.info("No jobs yet — start one above.")
    st.stop()

counts = {s: sum(1 for j in all_jobs if j.status == s) for s in STATUS_ICON}
cols = st.columns(5)
cols[0].metric("Jobs", len(all_jobs))
cols[1].metric("🔄 Running", counts[jobs.RUNNING] + counts[jobs.QUEUED])
cols[2].metric("⏸️ Needs you", counts[jobs.AWAITING])
cols[3].metric("✅ Done", counts[jobs.DONE])
cols[4].metric("❌ Failed", counts[jobs.FAILED])

live = any(j.is_active for j in all_jobs)
refresh = st.checkbox("Auto-refresh every 3s", value=live,
                      help="On by default while a worker is running.")

st.divider()


# ── Job list ─────────────────────────────────────────────────────────────────
def stage_bar(job: jobs.Job) -> str:
    """scout → lens → quill with each stage's state."""
    done = job.completed_stages
    parts = []
    for s in jobs.STAGES:
        if s in done:
            parts.append(f"✅ {s}")
        elif s == job.stage:
            parts.append(f"{STATUS_ICON.get(job.status, '·')} **{s}**")
        else:
            parts.append(f"· {s}")
    return "  →  ".join(parts)


st.subheader("All jobs")
for job in all_jobs:
    icon = STATUS_ICON.get(job.status, "·")
    label = f"{icon} `{job.id}`  ·  {job.question[:70]}{'…' if len(job.question) > 70 else ''}"
    with st.container(border=True):
        c1, c2 = st.columns([5, 2])
        c1.markdown(label)
        c1.caption(stage_bar(job))
        c2.caption(
            f"{job.model} · {time.strftime('%b %d %H:%M', time.localtime(job.created_at))}"
        )
        if c2.button("Open", key=f"open_{job.id}", use_container_width=True):
            st.session_state["job_id"] = job.id
            st.rerun()

# Default to the most recently touched job.
if "job_id" not in st.session_state or not jobs.get(st.session_state["job_id"]):
    st.session_state["job_id"] = max(all_jobs, key=lambda j: j.updated_at).id

job = jobs.get(st.session_state["job_id"])

st.divider()


# ── Detail ───────────────────────────────────────────────────────────────────
st.subheader(f"{STATUS_ICON.get(job.status, '·')} Job `{job.id}`")
st.markdown(f"**{job.question}**")
st.caption(f"{stage_bar(job)}  ·  model `{job.model}`"
           + ("  ·  auto-approve on" if job.auto_approve else ""))

if job.error:
    st.error(job.error)

# Controls: the approval gate, plus retry / cancel / delete.
ctl = st.columns(4)
if job.status == jobs.AWAITING:
    st.warning(GATE_PROMPT.get(job.stage, f"Approve and run {job.stage}?"))
    if ctl[0].button("✅ Approve", type="primary", key="approve", use_container_width=True):
        fresh = jobs.get(job.id)
        if fresh and fresh.status == jobs.AWAITING:     # ignore stale clicks
            jobs.approve(job.id)
            jobs.spawn_worker(job.id)
        st.rerun()
    if ctl[1].button("🚫 Stop here", key="reject", use_container_width=True):
        jobs.cancel(job.id)
        st.rerun()
elif job.status == jobs.FAILED:
    if ctl[0].button("🔁 Retry stage", type="primary", key="retry", use_container_width=True):
        jobs.approve(job.id)          # failed -> queued, same stage
        jobs.spawn_worker(job.id)
        st.rerun()
elif job.is_active:
    if ctl[0].button("🚫 Cancel", key="cancel", use_container_width=True):
        jobs.cancel(job.id)
        st.rerun()

if ctl[3].button("🗑️ Delete job", key="delete", use_container_width=True):
    jobs.delete(job.id)
    st.session_state.pop("job_id", None)
    st.rerun()


# ── Artifacts ────────────────────────────────────────────────────────────────
tab_plan, tab_results, tab_critique, tab_log = st.tabs(
    ["Research plan", "Experiment results", "Critique", "Worker log"]
)

with tab_plan:
    if job.plan_path and Path(job.plan_path).exists():
        st.caption(f"`{job.plan_path}`")
        st.markdown(Path(job.plan_path).read_text())
    else:
        st.caption("No plan yet — Scout hasn't finished.")

with tab_results:
    if job.bundle_path and Path(job.bundle_path).exists():
        bundle = json.loads(Path(job.bundle_path).read_text())
        st.metric("Experiments", f"{bundle['n_success']}/{bundle['n_total']} succeeded")
        for r in bundle.get("results", []):
            ok = r["status"] == "success"
            with st.expander(f"{'✅' if ok else '❌'} {r['name']} ({r['tool']})"):
                if not ok:
                    st.error(r.get("error", "unknown error"))
                    continue
                st.write(r.get("summary", "(no summary)"))
                for p in r.get("plot_paths", []):
                    if Path(p).exists():
                        st.image(p)
        st.caption(f"`{job.bundle_path}`")
    else:
        st.caption("No results yet — Lens hasn't finished.")

with tab_critique:
    if job.report_path and Path(job.report_path).exists():
        path = Path(job.report_path)
        st.caption(f"`{path}`")
        if path.suffix == ".md":
            st.markdown(path.read_text())
        else:
            st.json(json.loads(path.read_text()))
    else:
        st.caption("No critique yet — Quill hasn't finished.")

with tab_log:
    log = jobs.tail_log(job.id, n_lines=200)
    st.code(log or "(no output yet)", language="text")
    st.caption(f"`{job.log_path}` — last 200 lines")


# ── Auto-refresh ─────────────────────────────────────────────────────────────
# A fragment on a timer is the only piece that reruns on its own; it asks
# for a full-app rerun so every panel above picks up the worker's writes.
@st.fragment(run_every=3 if refresh else None)
def _ticker() -> None:
    if refresh:
        st.rerun(scope="app")


_ticker()
