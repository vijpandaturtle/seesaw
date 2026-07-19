"""Grader-validation fixtures — canned (inputs, expected verdict) pairs per grader.

    from eval.fixtures import load_all, check_offline
    print(check_offline())     # runs deterministic fixtures, reports mismatches + coverage

See loader.py for the fixture schema and the pass/fail verdict bands.
"""

from .loader import (
    FAIL_BAND,
    PASS_BAND,
    Fixture,
    FixtureError,
    band,
    check_llm,
    check_offline,
    human_review_payloads,
    load_all,
    resolve_grader,
    run_fixture,
)

__all__ = [
    "load_all", "check_offline", "check_llm", "human_review_payloads",
    "run_fixture", "resolve_grader", "band",
    "Fixture", "FixtureError", "PASS_BAND", "FAIL_BAND",
]
