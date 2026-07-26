"""Seesaw UI — pick a model, run the Scout -> Lens -> Quill pipeline, review each stage.

Run with:
    streamlit run ui/app.py

Streamlit reruns this whole script on every interaction, so the console-based
HITL checkpoints in orchestrator/src/hitl.py (which block on input()) don't
work here. Instead this app calls the three orchestrator clients directly
(run_scout, run_lens, run_quill) and uses buttons as the approval gates,
persisting progress in st.session_state between reruns.
"""

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.src.clients.lens_client import run_lens
from orchestrator.src.clients.quill_client import run_quill
from orchestrator.src.clients.scout_client import run_scout

MODEL_OPTIONS = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "pythia-70m",
    "pythia-160m",
    "pythia-410m",
    "gpt-neo-125m",
]

st.set_page_config(page_title="Seesaw", page_icon="🪶", layout="wide")


def reset() -> None:
    for key in ("plan_path", "plan_text", "bundle", "bundle_path", "report", "report_path"):
        st.session_state.pop(key, None)


def env_status(name: str) -> str:
    return "✅ set" if os.getenv(name) else "⚠️ missing"


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🪶 Seesaw")
    st.caption("Scout → Lens → Quill")

    st.subheader("Target model")
    model_choice = st.selectbox(
        "TransformerLens model for Lens experiments",
        MODEL_OPTIONS,
        index=0,
        help="Passed as a hint to Scout's research plan — Lens will use this "
        "unless the plan strongly implies otherwise.",
    )
    custom_model = st.text_input("...or type a custom model name", value="")
    selected_model = custom_model.strip() or model_choice

    st.divider()
    st.subheader("Environment")
    st.text(f"ANTHROPIC_API_KEY  {env_status('ANTHROPIC_API_KEY')}")
    st.text(f"FIRECRAWL_API_KEY  {env_status('FIRECRAWL_API_KEY')}")
    st.caption("FIRECRAWL_API_KEY is optional — search_arxiv still works without it.")

    st.divider()
    if st.button("Start over"):
        reset()
        st.rerun()

# ── Main ─────────────────────────────────────────────────────────────────────
st.header("Research question")
question = st.text_area(
    "What do you want to investigate?",
    placeholder="What attention heads mediate indirect object identification in GPT-2 Small?",
    height=80,
)

if st.button("1. Run Scout", type="primary", disabled=not question.strip()):
    reset()
    with st.spinner("Scout is searching arXiv and the web..."):
        try:
            plan_path = run_scout(question.strip())
            st.session_state["plan_path"] = plan_path
            st.session_state["plan_text"] = plan_path.read_text()
        except Exception as e:
            st.error(f"Scout failed: {e}")

# ── Stage 1 result: Research Plan ───────────────────────────────────────────
if "plan_text" in st.session_state:
    st.divider()
    st.header("Scout's Research Plan")
    with st.expander("Full plan", expanded=True):
        st.markdown(st.session_state["plan_text"])

    st.info(f"Target model for experiments: **{selected_model}**")

    if st.button("2. Approve plan and run Lens", type="primary"):
        with st.spinner(f"Lens is running experiments on {selected_model}..."):
            try:
                plan_with_model = (
                    st.session_state["plan_text"]
                    + f"\n\n**Target Model (required, overrides any other model mentioned above)**: {selected_model}\n"
                )
                bundle, bundle_path = run_lens(plan_with_model)
                st.session_state["bundle"] = bundle
                st.session_state["bundle_path"] = bundle_path
            except Exception as e:
                st.error(f"Lens failed: {e}")

# ── Stage 2 result: Experiment Results ──────────────────────────────────────
if "bundle" in st.session_state:
    st.divider()
    st.header("Lens Experiment Results")
    bundle = st.session_state["bundle"]
    st.metric("Experiments", f"{bundle['n_success']}/{bundle['n_total']} succeeded")

    for r in bundle["results"]:
        icon = "✅" if r["status"] == "success" else "❌"
        with st.expander(f"{icon} {r['name']} ({r['tool']})"):
            if r["status"] != "success":
                st.error(r.get("error", "unknown error"))
                continue
            st.write(r.get("summary", "(no summary)"))
            for p in r.get("plot_paths", []):
                if Path(p).exists():
                    st.image(p)

    if st.button("3. Send results to Quill for critique", type="primary"):
        with st.spinner("Quill is critiquing the results..."):
            try:
                report, report_path = run_quill(st.session_state["bundle_path"])
                st.session_state["report"] = report
                st.session_state["report_path"] = report_path
            except Exception as e:
                st.error(f"Quill failed: {e}")

# ── Stage 3 result: Critique Report ─────────────────────────────────────────
if "report" in st.session_state:
    st.divider()
    st.header("Quill's Critique")
    report = st.session_state["report"]

    badge = {"strong": "🟢", "moderate": "🟡", "weak": "🔴"}[report.overall_assessment]
    st.subheader(f"{badge} Overall: {report.overall_assessment.upper()}")
    st.write(report.overall_summary)
    st.caption(f"Coverage: {report.coverage_verdict}")

    st.subheader("Per-experiment critiques")
    for c in report.critiques:
        valid_icon = {"strong": "✅", "moderate": "⚠️", "weak": "❌"}[c.validity]
        with st.expander(f"{valid_icon} {c.experiment_name} — {c.validity}"):
            st.write("**Conclusions supported:**", "Yes" if c.conclusions_supported else "No")
            if c.issues:
                st.write("**Issues:**")
                for i in c.issues:
                    st.write(f"- {i}")
            if c.alternative_explanations:
                st.write("**Alternative explanations:**")
                for a in c.alternative_explanations:
                    st.write(f"- {a}")

    if report.gaps:
        st.subheader("Research gaps")
        for g in report.gaps:
            sev = {"critical": "🔴", "important": "🟡", "minor": "🟢"}[g.severity]
            st.write(f"{sev} **{g.description}** — {g.why_needed} (`{g.suggested_tool}`)")

    if report.followups:
        st.subheader("Suggested follow-up experiments")
        for f in report.followups:
            pri = {"high": "🔴", "medium": "🟡", "low": "🟢"}[f.priority]
            st.write(f"{pri} **{f.name}** (`{f.tool}`) — {f.rationale}")

    st.caption(f"Full report saved to `{st.session_state['report_path']}`")
