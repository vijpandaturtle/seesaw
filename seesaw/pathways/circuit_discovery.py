"""Circuit discovery: what mechanism does the model use for a logit-difference task like IOI?

Measure → nominate → verify → evaluate:

1. Measure — is there a behaviour? Clean logit diff must clear EFFECT_MIN.
2. Nominate — cheap passes that rank every head.
   - Direct logit attribution: which heads *write* toward the answer, and
     which write against it. Descriptive only.
   - Attribution patching (with corrupted prompts): every head's estimated
     patching effect from one backward pass. Without corrupted prompts, a full
     mean-ablation sweep takes its place, which is fine on small models only.
3. Verify — real interventions on the shortlist only. Head patching with
   corrupted prompts, mean ablation without. A head is confirmed when its
   effect alone clears CONFIRM_EFFECT.
4. Evaluate — mean-ablate everything outside the confirmed heads and see how
   much behaviour survives (faithfulness), against random circuits of the
   same size.
5. Wire (with corrupted prompts) — which connections carry it. Edge
   attribution patching (EAP-IG) ranks every sender → receiver edge; path
   patching measures the top EDGE_K, and an edge is confirmed when patching
   it alone moves the logit diff by at least EDGE_CONFIRM of the gap, in
   either direction. The result is a wiring diagram, not just a head list.

Head roles in `findings`:
- confirmed_writers — confirmed, and DLA says they write the answer (name movers in IOI).
- necessary_but_indirect — confirmed, but don't write it directly: upstream
  heads like S-inhibition and induction heads.
- writes_but_redundant — write the answer but fail verification alone:
  backup heads, which only matter once the primary ones are removed.
- writes_against_answer — DLA-negative heads (negative name movers in IOI).

Verdicts: no_effect · circuit_found · partial_circuit · no_circuit
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformer_lens import HookedTransformer

from seesaw.interp._helpers import parse_head, tokens_to_ids, top_heads
from seesaw.interp.causal_interventions import ablation, attribution_patching, head_patching
from seesaw.interp.circuit_finding import (
    circuit_evaluation, edge_attribution_patching, path_patching,
)
from seesaw.interp.metrics import logit_diff
from seesaw.interp.visualization import direct_logit_attribution

from .result import PathwayResult

# Clean logit diff below this (in logits) means there's no behaviour to explain.
EFFECT_MIN = 1.0
# Heads passed from nominate to verify.
SHORTLIST_K = 15
# DLA heads reported as writers / writing against.
DLA_K = 8
# A head is confirmed if, alone, patching it recovers (or ablating it removes)
# at least this fraction of the effect. 0.1 keeps IOI's S-inhibition heads,
# which matter individually at roughly 0.13-0.31 on GPT-2 small.
CONFIRM_EFFECT = 0.10
# Faithfulness needed for "circuit_found", and how far above random circuits it must sit.
FAITHFUL_MIN = 0.5
RANDOM_MARGIN = 0.25
# Edges ranked by EAP-IG and then path-patched, and the share of the gap one
# edge alone must move (either sign) to count as part of the wiring.
EDGE_K = 30
EDGE_CONFIRM = 0.05
EAP_IG_STEPS = 5


def run(
    model: HookedTransformer,
    prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    output_dir: Path,
    corrupted_prompts: list[str] | None = None,
    reference_prompts: list[str] | None = None,
) -> PathwayResult:
    """Run the circuit-discovery pathway on a logit-difference task.

    Args:
        model:             Loaded HookedTransformer.
        prompts:           Clean prompts.
        positive_tokens:   Correct answer token per prompt (e.g. " Mary").
        negative_tokens:   Distractor token per prompt (e.g. " John").
        output_dir:        Where every step saves its plots.
        corrupted_prompts: Position-aligned counterfactuals (e.g. IOI with S2
                           swapped). Enables attribution patching and head
                           patching; without them the pathway falls back to
                           ablation sweeps.
        reference_prompts: Where circuit evaluation's ablation means come from
                           (for IOI, the ABC distribution). Defaults to `prompts`.

    Returns:
        PathwayResult with every step, head roles, and a verdict.
    """
    result = PathwayResult(pathway="circuit_discovery")

    # ── 1. Measure ────────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(model.to_tokens(prompts))
    baseline = logit_diff(
        logits, tokens_to_ids(model, positive_tokens), tokens_to_ids(model, negative_tokens)
    ).mean().item()
    result.findings["baseline_logit_diff"] = round(baseline, 4)
    if baseline < EFFECT_MIN:
        return result.stop(
            "no_effect", f"clean logit diff {baseline:.2f} is below {EFFECT_MIN}, nothing to explain")
    result.evidence.append(f"The model prefers the answer by {baseline:.2f} logits on the clean prompts.")

    # ── 2. Nominate ───────────────────────────────────────────────────────────
    dla = direct_logit_attribution.compute(model, prompts, positive_tokens, negative_tokens, output_dir)
    result.steps["direct_logit_attribution"] = dla
    head_attrs = torch.tensor(dla.data["head_attrs"])
    writers = [h for h, _ in top_heads(head_attrs, DLA_K)]
    against = [h for h, _ in top_heads(-head_attrs, DLA_K)]

    if corrupted_prompts:
        atp = attribution_patching.compute(
            model, prompts, corrupted_prompts, positive_tokens, negative_tokens, output_dir)
        result.steps["attribution_patching"] = atp
        ranked = [h for h, _ in top_heads(torch.tensor(atp.data["head_effects"]), SHORTLIST_K)]
    else:
        sweep = ablation.compute(model, prompts, positive_tokens, negative_tokens, output_dir)
        result.steps["ablation_sweep"] = sweep
        result.evidence.append(
            "No corrupted prompts: heads were nominated and tested by mean ablation, which pushes "
            "activations off-distribution and misses redundant heads. Weaker evidence than patching.")
        ranked = [h for h, _ in top_heads(torch.tensor(sweep.data["effects"]), SHORTLIST_K)]
    # Writers are always tested too: a head can write the answer while
    # attribution ranks it low (saturation), and the roles need its verdict.
    shortlist = list(dict.fromkeys(ranked + writers))

    # ── 3. Verify ─────────────────────────────────────────────────────────────
    if corrupted_prompts:
        verify = head_patching.compute(
            model, prompts, corrupted_prompts, positive_tokens, negative_tokens, shortlist, output_dir)
        result.steps["head_patching"] = verify
        effect = verify.data["per_head"]
        how = "patching it alone recovers"
    else:
        verify = ablation.compute(
            model, prompts, positive_tokens, negative_tokens, output_dir, heads=shortlist)
        result.steps["ablation"] = verify
        effects = verify.data["effects"]
        effect = {h: effects[l][hd] / baseline for h in shortlist for l, hd in [parse_head(h)]}
        how = "ablating it alone removes"

    confirmed = [h for h in shortlist if effect[h] >= CONFIRM_EFFECT]
    result.findings.update({
        "confirmed_heads":        confirmed,
        "confirmed_writers":      [h for h in confirmed if h in writers],
        "necessary_but_indirect": [h for h in confirmed if h not in writers and h not in against],
        "writes_but_redundant":   [h for h in writers if h not in confirmed],
        "writes_against_answer":  against,
        "head_effects":           {h: round(effect[h], 4) for h in shortlist},
    })
    for h in sorted(confirmed, key=lambda h: -effect[h])[:5]:
        result.evidence.append(f"{h}: {how} {effect[h]:.0%} of the effect.")
    if corrupted_prompts:
        result.evidence.append(
            f"All {len(shortlist)} shortlisted heads patched together recover "
            f"{verify.data['joint_recovery']:.0%}.")
    if not confirmed:
        return result.stop("no_circuit", f"no shortlisted head reached {CONFIRM_EFFECT:.0%} alone")

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    ev = circuit_evaluation.compute(
        model, prompts, positive_tokens, negative_tokens, confirmed, output_dir,
        reference_prompts=reference_prompts)
    result.steps["circuit_evaluation"] = ev
    faith, rand = ev.data["faithfulness"], ev.data["random_faithfulness"]
    result.findings.update({
        "faithfulness": faith, "random_faithfulness": rand,
        "necessity": ev.data["necessity"], "random_necessity": ev.data["random_necessity"],
    })
    result.evidence.append(
        f"Keeping only the {len(confirmed)} confirmed heads preserves {faith:.0%} of the logit "
        f"diff; random sets of {len(confirmed)} heads preserve {rand:.0%}.")
    result.evidence.append(
        f"Ablating them removes {ev.data['necessity']:.0%} (random: {ev.data['random_necessity']:.0%}).")

    if faith >= FAITHFUL_MIN and faith - rand >= RANDOM_MARGIN:
        result.verdict = "circuit_found"
    else:
        result.verdict = "partial_circuit"

    # ── 5. Wire ───────────────────────────────────────────────────────────────
    if corrupted_prompts:
        eap = edge_attribution_patching.compute(
            model, prompts, corrupted_prompts, positive_tokens, negative_tokens, output_dir,
            top_k=EDGE_K, ig_steps=EAP_IG_STEPS)
        result.steps["edge_attribution_patching"] = eap
        pp = path_patching.compute(
            model, prompts, corrupted_prompts, positive_tokens, negative_tokens,
            [e for e, _ in eap.data["top_edges"]], output_dir)
        result.steps["path_patching"] = pp
        confirmed_edges = {e: r for e, r in pp.data["per_edge"].items() if abs(r) >= EDGE_CONFIRM}
        result.findings["confirmed_edges"] = dict(sorted(confirmed_edges.items(), key=lambda kv: -abs(kv[1])))
        between = {e: r for e, r in confirmed_edges.items()
                   if e.split("->")[0] in confirmed and e.split("->")[1].split(".")[0] in confirmed}
        result.evidence.append(
            f"{len(confirmed_edges)} of the top {len(pp.data['per_edge'])} edges by EAP-IG move the "
            f"logit diff by at least {EDGE_CONFIRM:.0%} when patched alone.")
        if between:
            top = sorted(between.items(), key=lambda kv: -abs(kv[1]))[:4]
            result.evidence.append(
                "Connections between confirmed heads: "
                + ", ".join(f"{e} {r:+.0%}" for e, r in top) + ".")
    return result
