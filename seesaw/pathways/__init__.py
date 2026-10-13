"""Hand-coded pathways — expert recipes that chain seesaw.interp tools to answer an interp question.

A single `seesaw.interp` function answers a narrow question ("which heads write
the answer direction?"). A pathway is the sequence a researcher would actually
run, where each step's output decides the next step's inputs: DLA nominates
heads, ablation checks they're load-bearing, patching localises where the
signal enters.

Pathways sit between the library and the agents:

    seesaw.interp      — single techniques, pure functions
    seesaw.pathways    — fixed, hand-written chains of those techniques  ← here
    lens               — an LLM choosing techniques on its own

They import only from `seesaw.interp`, never from Lens, MCP or LangGraph, so a
pathway is the expert baseline Lens can be evaluated against: run both on the
same task and compare findings.

Each pathway is a module exposing `run(model, ..., output_dir) -> PathwayResult`
and follows the same shape: measure (is there an effect?) → nominate (cheap
ranking of every candidate) → verify (real interventions on the shortlist) →
conclude (fixed thresholds, named as module constants, produce a verdict).

Built:
    circuit_discovery — what mechanism does the model use for task X?
    attribute_audit   — does the model use an attribute, and does removing it remove the behaviour?
    feature_explain   — which SAE features drive this output, and what are they?

Each is checked against a known answer on GPT-2 small in
tests/test_pathways_gpt2.py.
"""

from .result import PathwayResult

__all__ = ["PathwayResult"]
