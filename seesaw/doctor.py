"""`seesaw doctor`: is this machine ready to run a job?

Each check says what it found and, when something's wrong, what to do about
it, so a first job doesn't fail twenty minutes in on a missing token or a
model that doesn't fit. Nothing here costs money: the network checks are the
Anthropic model list and Hugging Face's access check, both free.
`--offline` skips them.

    seesaw doctor
    seesaw doctor --model google/gemma-2-2b    # check the models you'll use
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"
MARKS = {OK: "✓", WARN: "!", FAIL: "✗", SKIP: "·"}

# Hugging Face orgs whose models need a license accepted on the model's page.
GATED_ORGS = ("google/", "meta-llama/")
GiB = 1024 ** 3


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fix: str = ""


# ── individual checks ───────────────────────────────────────────────────────
def check_python(version: tuple = sys.version_info[:3]) -> Check:
    shown = ".".join(map(str, version))
    if (3, 11) <= tuple(version[:2]) <= (3, 12):
        return Check("Python", OK, shown)
    return Check("Python", FAIL, f"{shown}; seesaw needs 3.11 or 3.12",
                 "reinstall with: uv tool install --python 3.12 ...")


def check_anthropic(online: bool) -> Check:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return Check("Anthropic key", FAIL, "ANTHROPIC_API_KEY is not set",
                     "add ANTHROPIC_API_KEY=... to a .env in this folder (see .env.example)")
    if not online:
        return Check("Anthropic key", OK, "set (not checked: --offline)")
    try:
        import anthropic

        anthropic.Anthropic(api_key=key, timeout=10, max_retries=0).models.list(limit=1)
    except Exception as exc:                       # noqa: BLE001 — reported, not raised
        if type(exc).__name__ in ("AuthenticationError", "PermissionDeniedError"):
            return Check("Anthropic key", FAIL, "the key was rejected",
                         "create a new key at console.anthropic.com and update .env")
        return Check("Anthropic key", WARN, f"set, but couldn't reach the API ({type(exc).__name__})",
                     "check your network, then run seesaw doctor again")
    return Check("Anthropic key", OK, "set and accepted")


def check_firecrawl() -> Check:
    if os.environ.get("FIRECRAWL_API_KEY", "").strip():
        return Check("Web search", OK, "FIRECRAWL_API_KEY set (used with --search)")
    return Check("Web search", SKIP, "FIRECRAWL_API_KEY not set: --search uses arXiv only")


def lens_backend() -> str:
    return os.environ.get("SEESAW_LENS_BACKEND", "local").strip() or "local"


def check_torch() -> Check:
    """TransformerLens and torch, and which device they'll run on."""
    missing = [m for m in ("torch", "transformer_lens") if importlib.util.find_spec(m) is None]
    if missing:
        status = WARN if lens_backend() == "modal" else FAIL
        return Check("Experiments", status, f"not installed: {', '.join(missing)}",
                     'install the [experiments] extra, e.g. uv tool install "seesaw[experiments] @ git+..."')
    import torch

    if torch.cuda.is_available():
        return Check("Experiments", OK, f"torch {torch.__version__} on CUDA ({torch.cuda.get_device_name(0)})")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return Check("Experiments", OK, f"torch {torch.__version__} on Apple MPS")
    return Check("Experiments", WARN, f"torch {torch.__version__} on CPU only: jobs run, but slowly",
                 "use a GPU machine, or SEESAW_LENS_BACKEND=modal to run Lens on Modal")


def total_ram() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def hf_cache_dir() -> Path:
    try:
        from huggingface_hub import constants

        return Path(constants.HF_HUB_CACHE)
    except ImportError:
        return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def is_cached(model: str) -> bool:
    """Are the model's weights downloaded? Its config alone often is (a tokenizer
    load fetches it), so look for the weights: one file, or every shard an index names."""
    try:
        import json

        from huggingface_hub import try_to_load_from_cache

        def have(name: str) -> str | None:
            path = try_to_load_from_cache(model, name)
            return path if isinstance(path, str) else None

        index = have("model.safetensors.index.json")
        if index:
            shards = set(json.loads(Path(index).read_text())["weight_map"].values())
            return all(have(s) for s in shards)
        return have("model.safetensors") is not None
    except Exception:                              # noqa: BLE001
        return False


def check_model_fits(model: str, ram: int | None = None, free_disk: int | None = None,
                     cached: bool | None = None) -> Check:
    """Will the model load here? Weights download as bf16 and load in float32."""
    from shared.gpus import params_b

    name = f"Fits: {model}"
    if lens_backend() == "modal":
        return Check(name, SKIP, "Lens runs on Modal; nothing loads here")
    size = params_b(model)
    if size is None:
        return Check(name, SKIP, "size not in the name; not checked")
    need_ram = size * 1e9 * 4 * 1.3               # float32 weights plus activations
    need_disk = 0 if (is_cached(model) if cached is None else cached) else size * 1e9 * 2
    ram = total_ram() if ram is None else ram
    if free_disk is None:
        cache = hf_cache_dir()
        probe = next((p for p in (cache, *cache.parents) if p.exists()), Path.home())
        free_disk = shutil.disk_usage(probe).free
    problems, fixes = [], []
    if ram and need_ram > ram:
        problems.append(f"needs ~{need_ram / GiB:.0f} GB memory, this machine has {ram / GiB:.0f} GB")
        fixes.append("pick a smaller model (Qwen/Qwen2.5-0.5B) or run Lens on Modal")
    if need_disk > free_disk:
        problems.append(f"needs ~{need_disk / GiB:.0f} GB to download, {free_disk / GiB:.0f} GB free")
        fixes.append("free some disk, or point HF_HOME at a bigger drive")
    if problems:
        return Check(name, FAIL, "; ".join(problems), "; ".join(fixes))
    where = "already downloaded" if not need_disk else f"~{need_disk / GiB:.0f} GB download"
    found = f"~{need_ram / GiB:.0f} GB memory, {where}"
    if ram and need_ram > 0.75 * ram:
        return Check(name, WARN, f"{found}; tight on {ram / GiB:.0f} GB, close other apps",
                     "or pick a smaller model, or run Lens on Modal")
    return Check(name, OK, found)


def hf_token() -> str | None:
    """HF_TOKEN, or the token `huggingface-cli login` saved."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    try:
        from huggingface_hub import get_token

        return get_token()
    except ImportError:
        return None


def check_model_access(model: str, online: bool) -> Check:
    """Gated models need a Hugging Face token and the license accepted on the model's page."""
    name = f"Access: {model}"
    gated = model.startswith(GATED_ORGS)
    token = hf_token()
    page = f"https://huggingface.co/{model}"
    if gated and not token:
        return Check(name, FAIL, "gated on Hugging Face, and there's no token (HF_TOKEN or huggingface-cli login)",
                     f"accept the license at {page}, then add HF_TOKEN=... to .env")
    if not online:
        return Check(name, OK, "gated; token set (not checked: --offline)" if gated else "open model")
    try:
        from huggingface_hub import auth_check

        auth_check(model, token=token)
    except ImportError:
        return Check(name, SKIP, "huggingface_hub not installed; not checked")
    except Exception as exc:                       # noqa: BLE001
        kind = type(exc).__name__
        if kind == "GatedRepoError":
            return Check(name, FAIL, "your token hasn't been granted access",
                         f"accept the license at {page} (approval can take a while)")
        if kind == "RepositoryNotFoundError":
            return Check(name, FAIL, "no such model on Hugging Face, or the token can't see it",
                         "check the name against seesaw jobs submit --help")
        return Check(name, WARN, f"couldn't check ({kind})", "check your network")
    return Check(name, OK, "access granted" if gated else "open model")


def check_job_store() -> Check:
    from shared.db import jobs

    try:
        jobs.list_jobs(limit=1)
    except Exception as exc:                       # noqa: BLE001
        if jobs.is_postgres():
            return Check("Job store", FAIL, f"can't reach Postgres ({type(exc).__name__})",
                         "check DATABASE_URL, or unset it to use a local SQLite file")
        return Check("Job store", FAIL, f"can't open {jobs.db_path()} ({type(exc).__name__})",
                     "set SEESAW_DB_PATH to a writable file")
    if jobs.is_postgres():
        return Check("Job store", OK, "Postgres (DATABASE_URL): jobs show on the dashboard")
    return Check("Job store", OK, f"SQLite at {jobs.db_path()}")


def check_modal() -> Check:
    if lens_backend() != "modal":
        return Check("Modal", SKIP, "Lens runs on this machine (SEESAW_LENS_BACKEND unset)")
    if importlib.util.find_spec("modal") is None:
        return Check("Modal", FAIL, "SEESAW_LENS_BACKEND=modal but modal isn't installed",
                     "install the [modal] extra")
    if os.environ.get("MODAL_TOKEN_ID") or (Path.home() / ".modal.toml").exists():
        return Check("Modal", OK, f"token found; app {os.environ.get('SEESAW_MODAL_APP', 'seesaw-lens')}")
    return Check("Modal", FAIL, "no Modal token", "run: modal token new")


# ── report ───────────────────────────────────────────────────────────────────
def run_checks(models: list[str], online: bool = True) -> list[Check]:
    checks = [check_python(), check_anthropic(online), check_firecrawl(), check_torch()]
    for model in models:
        checks += [check_model_access(model, online), check_model_fits(model)]
    checks += [check_job_store(), check_modal()]
    return checks


def render(checks: list[Check]) -> str:
    width = max(len(c.name) for c in checks) + 2
    lines = []
    for c in checks:
        lines.append(f"  {MARKS[c.status]} {c.name:<{width}}{c.detail}")
        if c.fix and c.status in (FAIL, WARN):
            lines.append(f"    {'':<{width}}→ {c.fix}")
    fails = sum(c.status == FAIL for c in checks)
    warns = sum(c.status == WARN for c in checks)
    if fails:
        lines.append(f"\n{fails} problem(s) to fix before a job will run.")
    elif warns:
        lines.append(f"\nReady, with {warns} warning(s).")
    else:
        lines.append("\nReady. Try the first job in the README.")
    return "\n".join(lines)


def cmd_doctor(args, env_file: str | None = None) -> int:
    from shared.lens_tools import DEFAULT_MODEL

    from . import __version__

    models = args.model or [DEFAULT_MODEL]
    print(f"seesaw {__version__} · Python {sys.version.split()[0]} · "
          f".env: {env_file or 'none found'} · Lens: {lens_backend()}\n")
    checks = run_checks(models, online=not args.offline)
    print(render(checks))
    return 1 if any(c.status == FAIL for c in checks) else 0
