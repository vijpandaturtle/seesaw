"""Seesaw — a multi-agent framework for automated mechanistic interpretability research.

Public API (everything else in this repo is internal):

    import seesaw

    # Full pipeline (Scout → Lens → Quill, writes artifacts to */outputs/)
    seesaw.run_pipeline("What heads mediate IOI in GPT-2 Small?", skip_hitl=True)

    # Individual agents
    plan_path            = seesaw.run_scout("…question…")
    bundle, bundle_path  = seesaw.run_lens(plan_text)
    report, report_path  = seesaw.run_quill(bundle_path)

    # Direct experiment tools (loads the model on first use; needs [experiments] extra)
    from seesaw import experiments
    result = experiments.ablation("gpt2", prompts=[...],
                                  positive_tokens=[" Mary"], negative_tokens=[" John"])

    # Eval suite
    from seesaw import evals
    evals.check_offline()
    evals.run_task(evals.load_tasks()["quill_weak_bundle_1"])

Attributes are resolved lazily (PEP 562): `import seesaw` is fast and pulls in
neither torch nor langgraph until the corresponding function is first touched.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "run_pipeline", "run_scout", "run_lens", "run_quill",
    "experiments", "evals", "__version__",
]

_LAZY = {
    "run_pipeline": ("orchestrator.src.pipeline", "run_pipeline"),
    "run_scout": ("orchestrator.src.clients.scout_client", "run_scout"),
    "run_lens": ("orchestrator.src.clients.lens_client", "run_lens"),
    "run_quill": ("orchestrator.src.clients.quill_client", "run_quill"),
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    if name in ("experiments", "evals"):
        import importlib

        return importlib.import_module(f"seesaw.{name}")
    raise AttributeError(f"module 'seesaw' has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
