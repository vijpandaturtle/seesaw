# 2. Greater-Than Circuit
**Paper**: Conmy et al. (2023) — *Towards Automated Circuit Discovery* (ACDC)
**Model**: GPT-2 Small
**Mode**: `circuit_analysis`
**Input**: `"The war lasted from 1715 to 17"` (model predicts digits > 15)
**Metric**: Probability of valid year completions (> source year)

## What Lens should find

**Circuit location**: Concentrated in layers 4–8

**Attention Patterns:**
- Several heads in L4–L6 attend strongly from the `to 17` position back to `from 1715`
- These heads are tracking the "source year" token for comparison

**Ablation — key heads:**
- Ablating heads in L4–L6 that attend to the source year: valid year completion probability drops significantly (> 40%)
- Heads outside this range: minimal effect

**Activation Patching:**
- High recovery when patching `resid_post` at the source year token position in layers 4–7
- Patching at other token positions: low recovery

**Direct Logit Attribution:**
- Middle-layer heads (L4–L7) show positive attribution toward valid year tokens
- Early layers (0–3) show near-zero attribution

## Pass/Fail criteria
- ✅ PASS: Circuit concentrated in layers 4–8 (not uniformly distributed)
- ✅ PASS: Ablating 3–5 identified heads reduces valid year probability by > 40%
- ✅ PASS: Activation patching peaks at source year token position
- ❌ FAIL: No heads show significant ablation effect
- ❌ FAIL: Circuit spread uniformly across all 12 layers
