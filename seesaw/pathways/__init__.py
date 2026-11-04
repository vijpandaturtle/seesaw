"""Hand-coded pathways — expert recipes that chain Lens's tools to answer an interp question.

A single `seesaw.interp` function answers a narrow question ("which heads write
the answer direction?"). A pathway is the sequence a researcher would actually
run, where each step's output decides the next step's inputs: DLA nominates
heads, ablation checks they're load-bearing, patching localises where the
signal enters.

Pathways sit between the library and the agents:

    seesaw.interp      — single techniques, pure functions
    lens tool registry — those techniques as tools, run in-process or on Modal
    seesaw.pathways    — fixed, hand-written chains of those tools  ← here
    lens (the graph)   — an LLM choosing tools on its own

Every step is a call to a tool in Lens's registry, through
seesaw.pathways.tools: the same tools, run the same way (on the GPU app when
SEESAW_LENS_BACKEND=modal), as Lens's own experiments. Nothing else of Lens's
is used (no MCP, LangGraph or LLM), so a pathway is the expert baseline Lens
can be evaluated against: the same tools, chained by hand instead of by the
agent.

Each pathway is a module exposing `run(model, ..., output_dir) -> PathwayResult`,
where `model` is a loaded HookedTransformer or a model name,
and follows the same shape: measure (is there an effect?) → nominate (cheap
ranking of every candidate) → verify (real interventions on the shortlist) →
conclude (fixed thresholds, named as module constants, produce a verdict).

Built:
    circuit_discovery  — what mechanism does the model use for task X?
    concept_direction  — is a concept a direction the model uses, and can it be removed or steered?
                         (attribute_audit is its older name)
    probe_monitor      — would a probe on the activations make a good detector for X?
    spurious_audit     — does the answer depend on an attribute it shouldn't, and can it be erased?
    factual_recall     — where does the model retrieve a fact?
    editing_comparison — is the layer where a fact is traced the best layer to edit it?
    model_diffing      — what did a fine-tune change, and in which weights?
    feature_explain    — which SAE features drive this output, and what are they?

Each is checked against a known answer on GPT-2 small in
tests/test_pathways_gpt2.py.
"""

from .result import PathwayResult

__all__ = ["PathwayResult"]
