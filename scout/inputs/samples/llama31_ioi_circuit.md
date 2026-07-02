# Sample Research Question — Llama-3.1-8B IOI Circuit Analysis

## Model
`meta-llama/Meta-Llama-3.1-8B`

## Mode
`circuit_analysis`

## Prompt
```
When Mary and John went to the store, John gave a drink to
```

## Research Question

How does Llama-3.1-8B implement the Indirect Object Identification (IOI) task,
and is its circuit more efficient than GPT-2 Small's known 26-head implementation?

## Why This Question

GPT-2 Small solves IOI using a circuit of ~26 attention heads across 12 layers,
identified by Wang et al. (2022). Llama-3.1-8B has 32 layers, 32 heads, and 8B
parameters. If it solves the same task with fewer causally necessary heads — or
if the relevant heads emerge earlier in the network — that is a concrete, measurable
signal of architectural improvement.

This is the mech interp angle on "why is Llama-3.1-8B so good": not benchmark
scores, but circuit efficiency. A more specialised circuit means less redundancy
and more robust generalisation.

## Specific Hypotheses to Test

1. Llama-3.1-8B achieves a higher baseline logit diff on IOI than GPT-2 Small,
   reflecting greater model confidence in the correct answer.

2. The name-mover heads in Llama-3.1-8B are concentrated in later layers
   (proportionally equivalent to GPT-2's L9-L11), and fewer heads are causally
   necessary (< 26) to maintain the behaviour.

3. Patchscope probes on causally important heads will show semantic encoding of
   the IO token (Mary) by mid-network — earlier proportionally than GPT-2 Small.

## Expected Lens Experiments

1. **Ablation** — identify which heads are causally necessary (logit diff drop > threshold)
2. **Activation patching** — find which (layer, position) pairs carry IO information
3. **Direct logit attribution** — which heads write the IO token into the residual stream
4. **Attention pattern** — what does the top ablation-identified head attend to?
5. **Patchscope probe** — semantically decode the top name-mover heads' outputs

## Comparison Baseline

GPT-2 Small results from Wang et al. (2022):
- Baseline logit diff: ~3.5
- ~26 heads causally involved
- Name-mover heads at L9H9, L10H0, L9H6
- S-inhibition heads at L7H3, L7H9
- Induction heads at L5H5, L6H9

## References to Start With

- Wang et al. (2022) — "Interpretability in the Wild: a Circuit for IOI in GPT-2 Small"
- Ghandeharioun et al. (2024) — "Patchscopes: A Unifying Framework for Inspecting Hidden Representations"
- Dubey et al. (2024) — "The Llama 3 Herd of Models" (architecture details)
- Neel Nanda's IOI Colab — ground truth comparison baseline
