# Seesaw v0 — Validation Ground Truth

Reference findings from published papers. Lens output should be compared against these.
All four tasks run on GPT-2 Small unless noted.

---

## 1. Indirect Object Identification (IOI)
**Paper**: Wang et al. (2022) — *Interpretability in the Wild*  
**Model**: GPT-2 Small  
**Mode**: `circuit_analysis`  
**Input**: `"When Mary and John went to the store, John gave a drink to"`  
**Metric**: Logit diff = logit(IO token) − logit(Subject token)

### What Lens should find

**Baseline logit diff**: ~2.5–3.5 (clean run, GPT-2 Small)

**Direct Logit Attribution — top positive heads (write IO to output):**
| Head | Role | Expected DLA contribution |
|---|---|---|
| L9H6 | Name Mover | High positive |
| L9H9 | Name Mover | High positive |
| L10H0 | Name Mover | Moderate positive |
| L10H7 | Backup Name Mover | Moderate positive |
| L11H10 | Backup Name Mover | Moderate positive |

**Ablation — largest logit diff drop when zeroed:**
- L9H6, L9H9 ablation: logit diff drops by ~1.5–2.0
- L7H3, L7H9, L8H6, L8H10 (S-Inhibition): ablating these *increases* logit diff (suppressive)

**Activation Patching — high recovery positions:**
- Patching `resid_post` at the subject token (`John`) in layers 7–10 → recovery > 0.7
- Patching at the IO token (`Mary`) in layers 9–11 → recovery > 0.6
- Earlier layers (0–5): recovery < 0.2

**Attention Patterns:**
- L9H6, L9H9: strong attention from final token position → IO token (`Mary`)
- L0H1, L3H0 (Duplicate Token heads): attend from second `John` → first `John`

**Patchscopes (if implemented):**
- Probing `John` (subject) at layer 8: should decode as "repeated/duplicate name"
- Probing final position at layer 9: should decode as the IO name (`Mary`)

### Pass/Fail criteria
- ✅ PASS: L9H6 and L9H9 appear in top-5 DLA heads with positive contribution
- ✅ PASS: Ablating L9H6 + L9H9 reduces logit diff by > 30%
- ✅ PASS: Activation patching recovery peaks in layers 7–11 at subject/IO positions
- ❌ FAIL: If top DLA heads are in layers 0–5
- ❌ FAIL: If ablation effects are uniform across all heads (no clear circuit)

---

## 2. Greater-Than Circuit
**Paper**: Conmy et al. (2023) — *Towards Automated Circuit Discovery* (ACDC)  
**Model**: GPT-2 Small  
**Mode**: `circuit_analysis`  
**Input**: `"The war lasted from 1715 to 17"` (model predicts digits > 15)  
**Metric**: Probability of valid year completions (> source year)

### What Lens should find

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

### Pass/Fail criteria
- ✅ PASS: Circuit concentrated in layers 4–8 (not uniformly distributed)
- ✅ PASS: Ablating 3–5 identified heads reduces valid year probability by > 40%
- ✅ PASS: Activation patching peaks at source year token position
- ❌ FAIL: No heads show significant ablation effect
- ❌ FAIL: Circuit spread uniformly across all 12 layers

---

## 3. Gender Bias in Occupation Pronouns
**Paper**: Vig et al. (2020) — *Investigating Gender Bias via Causal Mediation Analysis*  
**Model**: GPT-2 (Small and Medium)  
**Mode**: `circuit_analysis`  
**Input**: `"The nurse said ___ was tired"` vs `"The engineer said ___ was tired"`  
**Metric**: Logit diff = logit("she") − logit("he") at the blank position

### What Lens should find

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

### Pass/Fail criteria
- ✅ PASS: Logit diff is significantly positive for "nurse", negative for "engineer"
- ✅ PASS: Ablating top 3–5 heads in layers 7–9 reduces |logit diff| by > 30%
- ✅ PASS: Activation patching at occupation token in layers 7–10 shows high recovery
- ✅ PASS: Control prompt ("teacher") shows near-zero logit diff
- ❌ FAIL: No significant logit diff between nurse/engineer prompts
- ❌ FAIL: No concentration of effect in specific heads

---

## 4. Factual Association (ROME)
**Paper**: Meng et al. (2022) — *Locating and Editing Factual Associations in GPT*  
**Model**: GPT-2 XL (this task requires XL — not Small)  
**Mode**: `factual_recall`  
**Input**: `"The Eiffel Tower is located in the city of"` → expects `" Paris"`  
**Metric**: Probability / rank of correct token at final position

### What Lens should find

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

### Pass/Fail criteria
- ✅ PASS: `Paris` emerges in logit lens between layers 13–17 in GPT-2 XL
- ✅ PASS: MLP layers 13–17 dominate DLA (not attention heads)
- ✅ PASS: Activation patching peaks at last subject token position in MLP layers
- ❌ FAIL: Fact emerges in first 5 layers (that would be token-frequency, not retrieval)
- ❌ FAIL: Attention heads dominate over MLPs (would contradict ROME finding)
- ❌ FAIL: Patching first subject token works better than last subject token

---

## Summary Table

| Task | Model | Key layers | Dominant component | Metric |
|---|---|---|---|---|
| IOI | GPT-2 Small | 7–11 | Attention heads | Logit diff IO−S |
| Greater-than | GPT-2 Small | 4–8 | Attention heads | Valid year probability |
| Gender bias | GPT-2 Small | 7–9 | Attention heads | Logit diff she−he |
| Factual recall | GPT-2 XL | 13–17 | MLP layers | Token rank / probability |

**Note**: The first three tasks are structurally similar (attention-head circuits, circuit_analysis mode).
Factual recall is deliberately different — it validates that Seesaw correctly identifies MLP-dominant mechanisms
and doesn't over-attribute everything to attention.
