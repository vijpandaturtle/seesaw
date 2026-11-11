"""`seesaw doctor`: each check finds the problem a tester would hit, and says how to fix it.

No network, no model loads: the checks that would call out are run offline
or with their inputs (memory, disk, token) passed in.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from seesaw import doctor
from seesaw.doctor import FAIL, OK, SKIP, WARN

GiB = doctor.GiB


@pytest.fixture(autouse=True)
def _local_lens(monkeypatch):
    monkeypatch.delenv("SEESAW_LENS_BACKEND", raising=False)


def test_python_versions():
    assert doctor.check_python((3, 12, 1)).status == OK
    assert doctor.check_python((3, 11, 0)).status == OK
    bad = doctor.check_python((3, 13, 0))
    assert bad.status == FAIL and bad.fix


def test_missing_anthropic_key_fails_with_a_fix(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    c = doctor.check_anthropic(online=False)
    assert c.status == FAIL and ".env" in c.fix
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    assert doctor.check_anthropic(online=False).status == OK


def test_gated_model_without_a_token_fails(monkeypatch):
    monkeypatch.setattr(doctor, "hf_token", lambda: None)
    c = doctor.check_model_access("google/gemma-2-2b", online=False)
    assert c.status == FAIL and "huggingface.co/google/gemma-2-2b" in c.fix


def test_open_model_needs_no_token(monkeypatch):
    monkeypatch.setattr(doctor, "hf_token", lambda: None)
    assert doctor.check_model_access("Qwen/Qwen2.5-1.5B", online=False).status == OK


def test_default_model_is_open():
    """A tester's first job runs without a Hugging Face token or license."""
    from shared.lens_tools import DEFAULT_MODEL, MODELS

    assert DEFAULT_MODEL in MODELS
    assert not DEFAULT_MODEL.startswith(doctor.GATED_ORGS)


@pytest.mark.parametrize("model, ram_gb, disk_gb, cached, status", [
    ("Qwen/Qwen2.5-1.5B", 32, 100, False, OK),
    ("Qwen/Qwen2.5-7B", 16, 100, False, FAIL),     # ~34 GB in float32
    ("Qwen/Qwen2.5-3B", 16, 100, False, WARN),     # ~15 GB: fits, barely
    ("Qwen/Qwen2.5-1.5B", 32, 1, False, FAIL),     # can't download
    ("Qwen/Qwen2.5-1.5B", 32, 1, True, OK),        # already downloaded
    ("gpt2", 32, 100, False, SKIP),                # no size in the name
])
def test_model_fits(model, ram_gb, disk_gb, cached, status):
    c = doctor.check_model_fits(model, ram=ram_gb * GiB, free_disk=disk_gb * GiB, cached=cached)
    assert c.status == status, c.detail
    if status == FAIL:
        assert c.fix


def test_nothing_loads_locally_on_modal(monkeypatch):
    monkeypatch.setenv("SEESAW_LENS_BACKEND", "modal")
    c = doctor.check_model_fits("Qwen/Qwen2.5-7B", ram=8 * GiB, free_disk=1 * GiB, cached=False)
    assert c.status == SKIP


def test_job_store(monkeypatch, tmp_path):
    monkeypatch.setenv("SEESAW_DB_PATH", str(tmp_path / "jobs.db"))
    assert doctor.check_job_store().status == OK
    (tmp_path / "a-file").write_text("")
    monkeypatch.setenv("SEESAW_DB_PATH", str(tmp_path / "a-file" / "jobs.db"))   # a file can't be a folder
    c = doctor.check_job_store()
    assert c.status == FAIL and "SEESAW_DB_PATH" in c.fix


def test_render_counts_problems():
    out = doctor.render([doctor.Check("A", OK, "fine"), doctor.Check("B", FAIL, "broken", "fix it")])
    assert "✗ B" in out and "→ fix it" in out and "1 problem(s)" in out
    assert "Ready" in doctor.render([doctor.Check("A", OK, "fine")])


def test_env_file_is_read_from_the_working_directory(tmp_path):
    """An installed seesaw lives in site-packages; the tester's .env is where they run it."""
    (tmp_path / ".env").write_text("SEESAW_DOCTOR_PROBE=from-cwd\n")
    code = ("import os; from seesaw.cli import _load_env; "
            "print(_load_env()); print(os.environ.get('SEESAW_DOCTOR_PROBE'))")
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True,
                         check=True).stdout.split()
    assert out == [str(tmp_path / ".env"), "from-cwd"]


def test_version_flag():
    out = subprocess.run([sys.executable, "-m", "seesaw.cli", "--version"], capture_output=True, text=True)
    assert out.stdout.startswith("seesaw ")
