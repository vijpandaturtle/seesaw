"""Running research jobs on Modal, and serving what they produce.

    modal deploy orchestrator/modal_app.py

This is the half of the system that Vercel cannot host: a stage takes minutes
to an hour, well past any serverless request limit. The dashboard inserts a
queued row and calls `run_stage.spawn(job_id)`, which returns immediately while
the work continues here — so there is no always-on machine and nothing to poll.

Scout, Quill, and Lens's graph run in this container. The five Lens experiment
tools do not: SEESAW_LENS_BACKEND=modal sends those to the GPU app in
lens/modal_app.py, so this one stays on CPU.

Artifacts go to a Volume rather than the container filesystem, which is
discarded when the function returns, and `artifacts` serves them back to the
dashboard over HTTP.
"""

from __future__ import annotations

import modal

APP_NAME = "seesaw-runner"
ARTIFACTS_DIR = "/artifacts"

image = (
    modal.Image.debian_slim(python_version="3.12")
    # Lens's graph runs here even though its experiments don't: run_experiment
    # imports TOOL_REGISTRY, and reading a tool's signature means importing the
    # module, which imports torch and matplotlib. CPU wheels only — the actual
    # forward passes happen on the GPU app, so this never touches a device.
    .pip_install(
        "torch",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "numpy",
        "matplotlib",
        "transformer-lens",
    )
    .pip_install(
        "python-dotenv",
        "pydantic",
        "langchain-core",
        "langchain-anthropic",
        "langchain-community",
        "langgraph",
        "langgraph-checkpoint-sqlite",
        "fastmcp",
        "mcp",
        "langchain-mcp-adapters",
        "pyyaml",
        "arxiv",
        "firecrawl-py",
        "psycopg[binary,pool]",
        "modal",
        "fastapi[standard]",
    )
    .env(
        {
            "SEESAW_ARTIFACTS_DIR": ARTIFACTS_DIR,
            # Experiments belong on the GPU app, not in here.
            "SEESAW_LENS_BACKEND": "modal",
            "MPLCONFIGDIR": "/tmp/mpl",
        }
    )
    .add_local_python_source("scout", "quill", "lens", "orchestrator", "shared")
)

artifacts = modal.Volume.from_name("seesaw-artifacts", create_if_missing=True)

# ANTHROPIC_API_KEY, DATABASE_URL, and optionally FIRECRAWL_API_KEY.
secrets = [modal.Secret.from_name("seesaw-secrets")]

app = modal.App(APP_NAME)


@app.function(
    image=image,
    volumes={ARTIFACTS_DIR: artifacts},
    secrets=secrets,
    # A pipeline stage is minutes to an hour; Scout and Quill are LLM-bound and
    # Lens waits on the GPU app.
    timeout=3600,
)
def run_stage(job_id: str) -> str:
    """Run whatever stage this job has queued, then return its status.

    Mirrors `python -m orchestrator.src.worker <job_id>`: claim atomically,
    run one stage, advance to the next gate. With auto_approve the worker
    continues through all three without returning.
    """
    from orchestrator.src.worker import run_job
    from shared.db import jobs

    print(f"▶ job {job_id}")
    run_job(job_id, worker_id=f"modal-{job_id}")
    artifacts.commit()          # make new files visible to the artifact server

    job = jobs.get(job_id)
    status = f"{job.status}@{job.stage}" if job else "missing"
    print(f"✔ job {job_id} → {status}")
    return status


@app.function(image=image, secrets=secrets)
@modal.fastapi_endpoint(method="POST")
def start(payload: dict):
    """Spawn a stage and return at once.

    `.spawn()` is what makes a serverless dashboard workable: the HTTP call
    finishes in milliseconds while the stage keeps running here for as long as
    it needs. Vercel could never hold that request open.

    Body: {"job_id": "...", "key": "..."}. The key is checked against
    RUNNER_KEY — an open trigger would let anyone spend the account's
    Anthropic and GPU credit. It travels in the body rather than a header so
    this module needs no FastAPI import locally, where it isn't installed.
    """
    import os

    from fastapi import HTTPException

    expected = os.getenv("RUNNER_KEY")
    if expected and (payload or {}).get("key") != expected:
        raise HTTPException(403, "bad or missing runner key")

    job_id = (payload or {}).get("job_id")
    if not job_id:
        raise HTTPException(400, "job_id is required")

    call = run_stage.spawn(job_id)
    return {"spawned": True, "job_id": job_id, "call_id": call.object_id}


@app.function(image=image, volumes={ARTIFACTS_DIR: artifacts}, secrets=secrets)
@modal.fastapi_endpoint(method="GET")
def artifact(path: str = "", job: str = "", kind: str = ""):
    """Serve one artifact from the volume.

    Addressed either by explicit relative path, or by (job, kind) which is
    resolved through the job store — the dashboard uses the latter, since it
    holds job ids rather than paths.
    """
    import os

    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    from shared.db import jobs

    artifacts.reload()          # pick up files another container just wrote

    if job and kind:
        record = jobs.get(job)
        column = {"plan": "plan_path", "bundle": "bundle_path", "report": "report_path"}
        attr = column.get(kind)
        if record is None or attr is None:
            raise HTTPException(404, "unknown job or kind")
        path = getattr(record, attr) or ""

    if not path:
        raise HTTPException(404, "no artifact")

    # Everything served must sit inside the volume, however it was addressed.
    root = os.path.realpath(ARTIFACTS_DIR)
    target = os.path.realpath(
        path if os.path.isabs(path) else os.path.join(ARTIFACTS_DIR, path)
    )
    if target != root and not target.startswith(root + os.sep):
        raise HTTPException(403, "outside the artifact volume")
    if not os.path.isfile(target):
        raise HTTPException(404, "artifact not on the volume")

    media = {
        ".png": "image/png",
        ".json": "application/json",
        ".md": "text/markdown; charset=utf-8",
    }.get(os.path.splitext(target)[1].lower(), "application/octet-stream")
    return FileResponse(target, media_type=media)


@app.local_entrypoint()
def main(job_id: str = "") -> None:
    """Run one job synchronously, for checking the deployment.

        modal run orchestrator/modal_app.py --job-id <id>
    """
    if not job_id:
        print("pass --job-id <id>; the dashboard creates them")
        return
    print(run_stage.remote(job_id))
