# 3. Gender Bias in Occupation Pronouns
**Paper**: Vig et al. (2020) — *Investigating Gender Bias via Causal Mediation Analysis*
**Model**: GPT-2 (Small and Medium)
**Mode**: `circuit_analysis`
**Input**: `"The nurse said ___ was tired"` vs `"The engineer said ___ was tired"`
**Metric**: Logit diff = logit("she") − logit("he") at the blank position

## What Lens should find

**Baseline logit diff:**
- "nurse" prompt: positive (model favours "she") — roughly +1.0 to +2.5
- "engineer" prompt: negative (model favours "he") — roughly −1.0 to −2.0
- "teacher" prompt (control): near zero

**Attention Patterns:**
- Heads in layers 7–9 attend from the pronoun/blank position back to the occupation word
- The occupation word attention weight correlates with the strength of the bias

**Ablation:**
- Ablating heads that attend occupation→pronoun in layers 7–9: logit diff moves toward zero
- Effect is stronger in GPT-2 Medium than Small

**Activation Patching:**
- Patching `resid_post` at the occupation token position in layers 7–10 produces high recovery
- Swapping "nurse" activations into "engineer" prompts: logit diff flips sign

**Patchscopes (key experiment):**
- Probe occupation token at layers 7–9
- Target prompt: `"Referring to this person, they are typically ___"`
- Expected decode: gender-associated words ("female", "woman") for "nurse" context
- This is what confirms the representation encodes gender, not just occupation

## Pass/Fail criteria
- ✅ PASS: Logit diff is significantly positive for "nurse", negative for "engineer"
- ✅ PASS: Ablating top 3–5 heads in layers 7–9 reduces |logit diff| by > 30%
- ✅ PASS: Activation patching at occupation token in layers 7–10 shows high recovery
- ✅ PASS: Control prompt ("teacher") shows near-zero logit diff
- ❌ FAIL: No significant logit diff between nurse/engineer prompts
- ❌ FAIL: No concentration of effect in specific heads
