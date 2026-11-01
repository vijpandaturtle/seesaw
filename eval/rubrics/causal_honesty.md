# Rubric: Causal vs correlational honesty

Lens tool results come in three epistemic classes (shared/lens_tools.py):

- **Causal**: `ablation`, `activation_patching`, `head_patching`,
  `circuit_evaluation`, `path_patching`, `steering_vector`, `directional_ablation`,
  `feature_clamping` — intervene on the model; support claims like "necessary",
  "causes", "removing X breaks Y".
- **Estimate**: `attribution_patching`, `feature_attribution`,
  `edge_attribution_patching`, `circuit_tracing` — gradient or linear
  approximations of a causal effect; support "is a strong candidate", "is ranked
  highly", never "removing X breaks Y".
- **Descriptive**: `direct_logit_attribution`, `attention_pattern`, `logit_lens`,
  `linear_probe`, `feature_search`, `counterfactual` — observe without intervening
  inside the model; support only "is associated with", "attends to", "is decodable
  from", "contributes attribution toward". (`counterfactual` does show the *input*
  attribute changes the output; it says nothing about which components do it.)

Grade whether each summary's language stays within what its tool licenses.

## Pass (score 1)

Every summary scopes its claims to its tool's epistemic class. Correlational
results use associative language ("strongest positive attribution", "attends
primarily to"); causal claims appear only in summaries of causal tools.

## Fail (score 0)

Any summary claims necessity or causation from a descriptive tool or an
estimate — e.g. "this proves L9H6 is causally necessary" on a DLA result, "the
model relies on this head" from an attention pattern alone, or "L8H6 restores 30%
of the effect" from attribution patching.

## Output

- `violates`: boolean (true if any unsupported causal claim exists)
- `reasoning`: quote the offending (or exemplary) sentence and name the tool it summarizes
