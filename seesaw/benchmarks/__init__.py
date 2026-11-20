"""Use-case benchmarks: does the interpretability method do the job better than the baseline without it?

A use case (seesaw/use_cases.py) names an outcome metric and a baseline that
uses no interpretability. A benchmark runs both arms on data whose answer is
known, over several models and seeds, and writes one row per (model, test set,
seed, arm) to outputs/bench/<use_case>.jsonl. No agents are involved: this
measures the method. Whether Scout, Lens and Quill reach the same verdict on
their own is the eval suite's job (eval/).

Each benchmark module has:

- DEFAULTS: the models, seeds and data it runs on unless told otherwise;
- run(models, seeds, ..., out_dir) -> list[dict]: the rows;
- summarize(rows) -> str: a Markdown table and a verdict per test set, against
  a win condition written in the module's docstring before anything was run.

`seesaw bench <use_case>` runs one.
"""

from __future__ import annotations

from importlib import import_module

# use case -> module in this package
BENCHMARKS = {
    "build_a_safety_monitor": "safety_monitor",
}


def names() -> list[str]:
    return list(BENCHMARKS)


def load(use_case: str):
    if use_case not in BENCHMARKS:
        raise ValueError(f"no benchmark for {use_case!r} yet; there are: {', '.join(BENCHMARKS)}")
    return import_module(f"{__name__}.{BENCHMARKS[use_case]}")
