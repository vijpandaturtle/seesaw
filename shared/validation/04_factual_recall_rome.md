# 4. Factual Association (ROME)
**Paper**: Meng et al. (2022) — *Locating and Editing Factual Associations in GPT*
**Model**: GPT-2 XL (this task requires XL — not Small)
**Mode**: `factual_recall`
**Input**: `"The Eiffel Tower is located in the city of"` → expects `" Paris"`
**Metric**: Probability / rank of correct token at final position

## What Lens should find

**Logit Lens:**
- Correct token (`Paris`) should emerge in top-5 predictions by layer 13–17
- Before layer 13: `Paris` rank is low (> 50)
- After layer 17: `Paris` rank stabilises in top-3
- Key signal: a sharp rank improvement at a specific layer — this is where the fact is "retrieved"

**Direct Logit Attribution:**
- MLP layers 13–17 show the highest positive attribution toward `Paris`
- Attention heads contribute less than MLPs for factual recall (opposite of IOI)
- This is the key difference from circuit_analysis mode — MLPs dominate here

**Activation Patching:**
- Corrupted prompt: `"The Eiffel Tower"` → replace with `"The Colosseum"` (wrong subject)
- Patching MLP output at the **last subject token** position in layers 13–17 recovers `Paris`
- Patching attention outputs: lower recovery than MLP patching
- Critical detail: patch at the **last token of the subject** (`Tower`), not the first (`The`)

## Pass/Fail criteria
- ✅ PASS: `Paris` emerges in logit lens between layers 13–17 in GPT-2 XL
- ✅ PASS: MLP layers 13–17 dominate DLA (not attention heads)
- ✅ PASS: Activation patching peaks at last subject token position in MLP layers
- ❌ FAIL: Fact emerges in first 5 layers (that would be token-frequency, not retrieval)
- ❌ FAIL: Attention heads dominate over MLPs (would contradict ROME finding)
- ❌ FAIL: Patching first subject token works better than last subject token
