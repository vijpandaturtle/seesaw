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

Each pathway is a module exposing `run(model, ..., output_dir) -> PathwayResult`.

Built: circuit_discovery.
"""

from .result import PathwayResult

__all__ = ["PathwayResult"]
