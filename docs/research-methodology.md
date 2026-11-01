# Seesaw research methodology: evidence tiers

**Status:** draft for discussion · **Used by:** the Model Atlas spec in seesaw-web
(`docs/model-atlas-spec.md`), whose datasheet `tier` field this defines · **Touches:** Scout's plan template, Lens's run order, Quill's
critique, `shared/lens_tools.py`, the eval suite

## Why

Seesaw has the parts of a methodology but not the methodology. The parts are good:

- `shared/lens_tools.py` sorts every Lens tool into an evidence kind (**descriptive**, **estimate**,
  **causal**) and lists nominate-then-verify chains per backend.
- Scout's prompt says "a strong plan nominates with a cheap tool and verifies with a causal one".
- Quill's prompts warn that attention isn't causation and that estimates aren't effects.
- The eval suite checks some of it: `causal_correlational_honesty` grades Lens summaries,
  `ioi_circuit` requires an ablation, and `lens_causal_language_trap` withholds causal tools on
  purpose.

What's missing is structure. All of the above is prose in prompts, so nothing holds a job to it:

1. **A plan is a flat list of experiments.** Nothing records that experiment 3 verifies what
   experiment 1 nominated, or that a hypothesis needs causal evidence to be answered.
2. **There is no claim object.** Hypotheses are bullet points in the plan, `hypothesis_tested` is
   free text on each experiment, and Lens's summaries make claims nobody tracks.
3. **Quill judges experiments, not claims.** `ExperimentCritique` grades each experiment's validity;
   nothing says "the central claim reached descriptive evidence but needed causal evidence".
4. **Enforcement is one LLM grader on Lens**, and it builds its tool lists from the `seesaw`
   backend even when the lever backend is active.
5. **Controls are a suggestion.** Some causal tools compare against random baselines
   (`circuit_evaluation`, `directional_ablation`, `feature_clamping`); others don't, and nothing
   requires one before a claim about a mechanism.

The proposal: make the evidence kinds into **tiers**, attach a tier to every claim, and have each
agent work in those terms. Most of the vocabulary already exists; this formalises it.

## The rule

> **A claim may not be stronger than the tier of its evidence.**

Everything else follows from this rule.

## Tiers

| Tier | Name | Evidence | Lets you say | Seesaw tools (text) | lever tools (vision) |
|---|---|---|---|---|---|
| 0 | Prior | Literature, Scout's search | "Prior work reports …" | (Scout) | (Scout) |
| 1 | Descriptive | Observation, no intervention | "represents", "attends to", "is decodable from", "contributes directly" | `logit_lens`, `attention_pattern`, `direct_logit_attribution`, `linear_probe`, `feature_search`, `counterfactual` | `logit_lens`, `clip_direct_effects` |
| 2 | Estimate | Gradient approximation of an intervention | "is a candidate", "ranks highest by estimated effect" | `attribution_patching`, `feature_attribution`, `edge_attribution_patching`, `circuit_tracing` | none yet |
| 3 | Causal | Intervention on the named components | "is necessary / sufficient for this behaviour on these prompts" | `ablation`, `activation_patching`, `head_patching`, `path_patching`, `steering_vector`, `directional_ablation`, `feature_clamping` | `vision_ablation`, `image_token_ablation`, `activation_patching`, `clip_head_ablation` |
| 4 | Controlled | Causal, beaten against a baseline, holding across prompt variations | "mediates", "is the mechanism / circuit for" | `circuit_evaluation`; tier 3 tools when run against random baselines and on varied prompt sets | none yet |

Notes:

- Tiers 1–3 are the existing kinds, renamed. A tool's tier comes from `shared/lens_tools.py`, read
  by code. An LLM never assigns it.
- **Tier 4 is new, and it's a requirement, not a tool list.** A causal result reaches it when it's
  compared with a matched random baseline and holds across prompt variations (the existing
  `robustness` grader already measures "pass^k across name pairs"). Some tools build the baseline
  in; for the others, Lens runs it as a control.
- `counterfactual` changes the input, not the model, so it stays descriptive, as the catalog has it.
- The vision backend has gaps at tiers 2 and 4. That's worth knowing; tiers don't fill it.

## The research loop

A job moves up the tiers in phases. Each phase exists to make the next one cheaper or better aimed.

```
Orient (T0–1)  →  Nominate (T1–2)  →  Verify (T3)  →  Control (T4)
 what's known     which components     are they        is it more than
 and visible      are candidates       necessary?      any random set?
```

1. **Each hypothesis declares its target tier.** "Which heads mediate IOI?" targets tier 4.
   "Is gender linearly decodable in GPT-2's residual stream?" targets tier 1, and stops there
   legitimately.
2. **Every causal experiment names what it verifies**: an earlier experiment's output, or a
   paper. Literature counts as a nomination. If Wang et al. already named the name-mover heads,
   Scout can go straight to head patching and cite them.
3. **Skipping a phase needs a stated reason**, which Quill checks. This stops the tiers becoming
   red tape: a question that needs one targeted ablation shouldn't have to run four phases first.
4. **A job ends when every hypothesis has reached its target tier, or is reported as stuck below
   it**, with the reason (no tool for it, budget, a negative result).

The chains already in `TOOL_CHAINS` become the default paths through the phases:

| Question shape (text) | Nominate | Verify | Control |
|---|---|---|---|
| Which heads do X | `attribution_patching` / `direct_logit_attribution` | `head_patching`, `ablation` with `heads` | `circuit_evaluation` |
| Which connections carry X | `edge_attribution_patching` | `path_patching` | random-edge baseline |
| Does the model use attribute A | `counterfactual` → `linear_probe` | `directional_ablation` / `steering_vector` | random directions (built in) |
| Which features do X | `feature_attribution` | `feature_clamping` | random features (built in) |
| **Vision:** does the answer use the image | `logit_lens` (description) | `vision_ablation` → `image_token_ablation` → `activation_patching` | none yet |

## What each agent does differently

**Scout** writes plans in tier terms. In the plan template:

- Each hypothesis gets an id, a target tier, and the claim it would license, written in that
  tier's language.
- Experiments are grouped by phase. Each names its hypothesis, its phase, and for tier 3 and up,
  what it verifies (`verifies: E1` or a citation).
- Skipped phases are listed, with reasons.

**Lens** runs the phases in order, and feeds each phase's output into the next: the top heads from
attribution patching become the `heads` for head patching. `check_followup` already does a version
of this for single follow-ups; this makes it the normal path, not an exception. Summaries use the
vocabulary of the tier their tool reached.

**Quill** critiques claims as well as experiments. For each hypothesis it reports the claim, the
target tier, the tier the evidence actually reached, and whether the conclusion overreaches. Gaps
become "claim H2 is at tier 2 and needs tier 3", and the follow-ups it generates are the next step
up the chain. Its existing confound and validity checks stay as they are.

**Approval gates** stay where they are for now (after Scout, before Quill). An option for later:
gate before tier 3 and 4 phases instead, since that's where GPU cost grows.

## Data model

Small additions; the tier itself is always computed from the tool, never stored by an LLM.

```python
# shared/lens_tools.py
TIER = {"descriptive": 1, "estimate": 2, "causal": 3}   # tier 4 is earned by controls, see above

# lens ExperimentSpec / ExperimentSpecModel: new fields
hypothesis_id: str           # "H1"
phase: Literal["orient", "nominate", "verify", "control"]
verifies: str | None         # experiment name or citation; required for verify/control

# quill CritiqueReport: new field
claims: list[ClaimAssessment]

class ClaimAssessment(BaseModel):
    hypothesis_id: str
    statement: str            # the claim as the job ends up making it
    target_tier: int
    evidence_tier: int        # computed from the tools of the experiments that support it
    experiments: list[str]
    verdict: Literal["supported", "overreaches", "stuck"]
    note: str
```

## How we'll know it helps

Use the existing eval suite, before and after, against `eval/baselines.json`.

- **New CODE graders**:
  - `scout.plan_follows_tiers`: every verify/control experiment names what it verifies, and every
    hypothesis targeting tier 3 or above has a causal experiment.
  - `quill.claim_tier_calibration`: on the planted weak/strong bundles, `evidence_tier` matches the
    tools actually run, and over-claims are flagged.
- **Fix** `causal_correlational_honesty` to read the active backend's catalog.
- **Rerun**: `ioi_circuit`, `lens_causal_language_trap`, `quill_weak_bundle`, `quill_strong_bundle`,
  `gender_bias`, `successor_heads`.

If plans and critiques don't measurably improve, that's the cheapest point to find out.

## Rollout

1. **Prompts only.** Tier table and rule in Scout's template and Quill's rubrics, generated from
   `shared/lens_tools.py` like the tool table is today. Add the graders and measure.
2. **Schema.** Add `hypothesis_id` / `phase` / `verifies` to specs and `claims` to the critique.
   The web job page can then show each hypothesis with its tier reached.
3. **Phased Lens runs.** Nomination outputs feed verification inputs automatically.
4. **Survey mode.** A job with a model and no question: run the standard tier-1 (and cheap tier-3)
   battery and write `datasheet.json`. This is the Model Atlas's producer. A datasheet section's
   `tier` becomes this evidence tier, which the Atlas already groups by.

## Effect on the Model Atlas spec

- Its "tiered methodology" line refers to this document. Until survey mode exists, datasheets stay
  hand-made or examples.
- `tier` in `datasheet.json` gets a defined meaning: evidence tier 1–4, as above.
- The model reference (a resolved hub id plus revision on every job) becomes a prerequisite for
  reusing tier results across jobs, since a result is only reusable under the same provenance.

## Open questions

1. Is tier 4 a separate tier, or a "controlled" flag on tier-3 results? A flag is simpler; a tier
   reads better in plans.
2. Does a literature nomination count as tier 0 evidence, or only as permission to skip a phase?
3. What counts as "holds across prompt variations": a fixed k, or per task as the eval tasks do now?
4. Vision has no tier 2 and no controls. Add attribution patching and a random-head baseline for
   CLIP heads to lever's tools, or accept a shorter ladder for vision models?
5. Should gates move to tier boundaries (step 3), or stay per agent?
