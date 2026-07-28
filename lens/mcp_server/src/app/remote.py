"""Running experiments on Modal instead of in this process.

Selected with SEESAW_LENS_BACKEND=modal (default: local), so nothing changes
until it's set. See lens/modal_app.py for the other side.

The remote call returns plot bytes rather than paths, because the container's
filesystem is gone the moment it finishes. Those bytes get written into the
usual PLOTS_DIR here, so everything downstream — the bundle, Quill, the
dashboard — sees exactly what a local run produces.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..config import PLOTS_DIR
from ..models.schemas import ExperimentResult

MODAL_APP = os.environ.get("SEESAW_MODAL_APP", "seesaw-lens")
MODAL_CLS = "LensRunner"


def backend() -> str:
    """'modal' or 'local'."""
    return (os.environ.get("SEESAW_LENS_BACKEND") or "local").strip().lower()


def is_remote() -> bool:
    return backend() == "modal"


def run_experiment_remote(spec: dict, output_dir: Path = PLOTS_DIR) -> ExperimentResult:
    """Execute one experiment spec on Modal and rebuild the local result.

    Args:
        spec: {tool, model_name, prompts, tool_kwargs, name}.
        output_dir: Where returned plots are written.

    Returns:
        An ExperimentResult indistinguishable from a local run's. Transport
        failures come back as status="failed" rather than raising, so a network
        problem costs one experiment instead of the whole job.
    """
    try:
        import modal
    except ImportError:
        return _failed(spec, "SEESAW_LENS_BACKEND=modal but the modal package "
                             "isn't installed (pip install modal)")

    try:
        runner_cls = modal.Cls.from_name(MODAL_APP, MODAL_CLS)
        payload = runner_cls().run_experiment.remote(spec)
    except Exception as exc:                        # noqa: BLE001 — reported as a result
        return _failed(spec, f"remote call failed: {type(exc).__name__}: {exc}")

    data = payload.get("result") or {}
    plots = payload.get("plots") or {}

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, blob in plots.items():
        # Only the basename crosses the wire; keep it that way so a remote
        # filename can't be used to write outside output_dir.
        target = output_dir / Path(filename).name
        target.write_bytes(blob)
        written.append(target)

    return ExperimentResult(
        name=data.get("name", spec.get("name", "experiment")),
        tool=data.get("tool", spec.get("tool", "unknown")),
        model_name=data.get("model_name", spec.get("model_name", "")),
        prompts=data.get("prompts") or spec.get("prompts") or [],
        summary=data.get("summary", ""),
        plot_paths=written,
        data=data.get("data") or {},
        status=data.get("status", "failed"),
        error=data.get("error"),
    )


def _failed(spec: dict, error: str) -> ExperimentResult:
    return ExperimentResult(
        name=spec.get("name", "experiment"),
        tool=spec.get("tool", "unknown"),
        model_name=spec.get("model_name", ""),
        prompts=spec.get("prompts") or [],
        status="failed",
        error=error,
    )
