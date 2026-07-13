# 8. Docstring Circuit
**Paper**: Heimersheim & Janiak (2023) — *A Circuit for Python Docstrings in a 4-Layer Attention-Only Transformer*
**Model**: attention-only-4l (a small TransformerLens-native toy model trained specifically for this paper — not GPT-2)
**Mode**: `circuit_analysis`
**Input**: Python docstring completion prompts, e.g. a function signature followed by a partial `:param` docstring line, where the model must predict the correct parameter name
**Metric**: Logit of the correct parameter name vs. other parameter names in the function signature

## What Lens should find

**Full circuit, all 4 layers**: because the model only has 4 layers, this is a good stress test for whether Lens's causal tools converge on a *complete* circuit rather than a partial one. The paper documents the full circuit precisely: **8 attention heads composing across all 4 layers, connected by 37 edges** between inputs, output, and the heads themselves. Earlier heads identify which parameter names have already been documented, a duplicate-token-style mechanism tracks position within the parameter list, and late heads move the correct (undocumented) parameter name to the output.

**Ablation:**
- Ablating any single head in the documented 8-head circuit should measurably reduce correct parameter-name prediction — with only 4 layers, there's less redundancy than in GPT-2 Small, so effects should be sharper and easier to attribute unambiguously

**Activation Patching:**
- Patching should cleanly localise to the specific token positions holding each parameter name, since the model is small enough that noise from unrelated computation is minimal

## Pass/Fail criteria
- ✅ PASS: Ablation identifies on the order of 8 heads spread across all 4 layers (matching the paper's documented circuit), not concentrated in just the final layer
- ✅ PASS: Activation patching localises cleanly to parameter-name token positions
- ❌ FAIL: All causal effect concentrates in a single layer (would suggest Lens is missing distributed contributions)
- ❌ FAIL: No clear localisation — effects are spread diffusely across all positions

**Note**: this task deliberately uses a non-GPT-2 model to validate that Lens's tools are model-agnostic, not hardcoded to GPT-2 Small's architecture or layer count.
