"""Lens experiment execution on Modal.

    modal deploy lens/modal_app.py

Only the five experiment tools run here — the forward passes. Lens's graph
(plan parsing, result interpretation, follow-up decisions) stays on the machine
running the worker, because that part is LLM calls and costs nothing to keep
local. So this container needs no API keys.

The boundary is the experiment spec, not the tool call: a HookedTransformer
can't cross the wire, so the caller sends {tool, model_name, prompts,
tool_kwargs} and gets back the result plus the plot bytes. The caller writes
those PNGs to its own outputs/, which is why bundles, the dashboard, and the
critique path need no changes at all.

Weights live on a Volume so a cold start reads them from Modal's disk instead
of re-downloading from HuggingFace, and a warm container keeps loaded models in
process — the same role _MODEL_CACHE plays locally.
"""

from __future__ import annotations

import modal

APP_NAME = "seesaw-lens"

# gpt2-scale work is comfortable on an A10G. Larger models want more memory for
# TransformerLens activation caching than for the weights themselves — see
# MAX_SWEEP_ROWS in app/helpers.py for the knob that bounds it.
GPU = "A10G"

CACHE_DIR = "/cache"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "numpy",
        "matplotlib",
        "transformer-lens",
        # Pulled in by the import chain rather than by the tools themselves:
        # config/settings.py calls load_dotenv, models/schemas.py uses pydantic.
        "python-dotenv",
        "pydantic",
    )
    .env({"HF_HOME": CACHE_DIR, "MPLCONFIGDIR": "/tmp/mpl"})
    # The tools and their helpers, without the agent/LLM layer.
    .add_local_python_source("lens")
)

volume = modal.Volume.from_name("seesaw-hf-cache", create_if_missing=True)

app = modal.App(APP_NAME)


@app.cls(
    image=image,
    gpu=GPU,
    volumes={CACHE_DIR: volume},
    # One job runs several experiments back to back; keeping the container warm
    # between them avoids reloading weights each time.
    scaledown_window=600,
    # A wedged sweep bills until something stops it. Locally the thread-based
    # sandbox cannot kill one; here the platform can.
    timeout=3600,
)
class LensRunner:
    @modal.enter()
    def setup(self) -> None:
        # Per-container model cache. Weights are the expensive part, so a warm
        # container answering a second experiment on the same model is cheap.
        self._models: dict = {}

    def _model(self, model_name: str):
        from transformer_lens import HookedTransformer

        from lens.mcp_server.src.config.settings import (
            TL_CENTER_UNEMBED,
            TL_CENTER_WRITING_WEIGHTS,
            TL_FOLD_LN,
            TL_REFACTOR_FACTORED_ATTN,
        )

        if model_name not in self._models:
            print(f"loading {model_name}")
            model = HookedTransformer.from_pretrained(
                model_name,
                center_writing_weights=TL_CENTER_WRITING_WEIGHTS,
                center_unembed=TL_CENTER_UNEMBED,
                fold_ln=TL_FOLD_LN,
                refactor_factored_attn_matrices=TL_REFACTOR_FACTORED_ATTN,
            )
            model.eval()
            self._models[model_name] = model
            volume.commit()          # persist anything newly downloaded
        return self._models[model_name]

    @modal.method()
    def run_experiment(self, spec: dict) -> dict:
        """Run one experiment and return its result plus plot bytes.

        Args:
            spec: {tool, model_name, prompts, tool_kwargs, name} — the same
                shape the local dispatcher builds.

        Returns:
            {"result": <ExperimentResult as a dict, plot_paths as bare
            filenames>, "plots": {filename: png bytes}}. Failures come back
            with status="failed" rather than raising, so one bad experiment
            doesn't abort the job.
        """
        import dataclasses
        import tempfile
        import traceback
        from pathlib import Path

        from lens.mcp_server.src.models.schemas import ExperimentResult
        from lens.mcp_server.src.tools import TOOL_REGISTRY, normalize_tool_kwargs

        name = spec.get("name", spec.get("tool", "experiment"))
        tool = spec.get("tool")
        model_name = spec.get("model_name", "gpt2")
        prompts = spec.get("prompts") or []

        def failed(error: str) -> dict:
            return {
                "result": dataclasses.asdict(
                    ExperimentResult(
                        name=name, tool=tool or "unknown", model_name=model_name,
                        prompts=prompts, status="failed", error=error,
                    )
                ),
                "plots": {},
            }

        tool_fn = TOOL_REGISTRY.get(tool)
        if tool_fn is None:
            return failed(f"Tool {tool!r} not in registry: {sorted(TOOL_REGISTRY)}")

        kwargs, missing, dropped = normalize_tool_kwargs(
            tool, spec.get("tool_kwargs") or {}, len(prompts)
        )
        if missing:
            return failed(
                f"Experiment spec for {tool!r} is missing required tool_kwargs "
                f"{missing}; got {sorted(spec.get('tool_kwargs') or {})}"
            )
        if dropped:
            print(f"{tool} doesn't accept {dropped} — ignoring")

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            try:
                result = tool_fn(
                    model=self._model(model_name),
                    prompts=prompts,
                    output_dir=out_dir,
                    **kwargs,
                )
            except Exception as exc:                 # noqa: BLE001 — reported back
                traceback.print_exc()
                return failed(f"{type(exc).__name__}: {exc}")

            plots = {}
            for path in result.plot_paths:
                path = Path(path)
                if path.exists():
                    plots[path.name] = path.read_bytes()

            payload = dataclasses.asdict(result)
            # Send bare filenames; the caller decides where they live on disk.
            payload["plot_paths"] = list(plots)
            return {"result": payload, "plots": plots}


@app.local_entrypoint()
def main(model_name: str = "gpt2") -> None:
    """Smoke test: run one ablation remotely and report what came back.

        modal run lens/modal_app.py
    """
    spec = {
        "name": "Ablation smoke test",
        "tool": "ablation",
        "model_name": model_name,
        "prompts": [
            "When John and Mary went to the store, John gave a drink to",
            "When Alice and Bob went to the park, Alice gave a ball to",
        ],
        "tool_kwargs": {
            "positive_tokens": [" Mary", " Bob"],
            "negative_tokens": [" John", " Alice"],
        },
    }
    out = LensRunner().run_experiment.remote(spec)
    result = out["result"]
    print(f"status : {result['status']}")
    print(f"error  : {result.get('error')}")
    print(f"plots  : {list(out['plots'])}")
    if result["status"] == "success":
        print(f"baseline_ld: {result['data'].get('baseline_ld')}")
        print(f"top_heads  : {result['data'].get('top_heads')}")
