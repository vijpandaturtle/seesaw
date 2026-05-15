# Sample Research Question — Indirect Object Identification (IOI)

## Research Question

How does GPT-2 Small implement the Indirect Object Identification (IOI) task?
Which attention heads and circuits are responsible, and what are the key computational mechanisms?

## Background

The IOI task is a classic mechanistic interpretability benchmark. Given a sentence like:

> "When Mary and John went to the store, John gave a drink to ___"

The model should predict "Mary" (the indirect object) rather than "John" (the subject).

This task was studied in depth by Wang et al. (2022) in "Interpretability in the Wild",
which identified a circuit of ~26 attention heads responsible for the behaviour in GPT-2 Small.

## Why This Question

- Well-studied — good for validating the Seesaw pipeline end-to-end
- Has a known ground truth circuit to compare against
- Experiments are well-defined and reproducible with TransformerLens
- GPT-2 Small runs on CPU — fast iteration

## Expected Lens Experiments

1. Attention pattern analysis on key heads
2. Activation patching to identify causal components
3. Ablation studies on name-mover, negative, and backup heads
4. Direct logit attribution to trace information flow
5. Logit lens to see where the model "decides" on the answer

## References to Start With

- Wang et al. (2022) — "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small"
- Elhage et al. (2021) — "A Mathematical Framework for Transformer Circuits"
- Neel Nanda's TransformerLens documentation and IOI Colab notebook
