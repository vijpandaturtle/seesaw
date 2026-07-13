# 5. Induction Heads
**Paper**: Olsson et al. (2022) — *In-context Learning and Induction Heads*
**Model**: GPT-2 Small
**Mode**: `circuit_analysis`
**Input**: A repeated random token sequence, e.g. `"...x1 x2 x3 x4... x1 x2 x3"` — the model should predict `x4` again on the second occurrence of `x1 x2 x3`.
**Metric**: Prefix-matching score (attention from the current token back to the token *after* the last occurrence of the same token) and the resulting boost to the correct next-token logit.

## What Lens should find

**Attention Patterns:**
- A small set of heads in the middle layers (commonly cited around L5–L6 in GPT-2 Small) show a distinctive "prefix matching" pattern: strong attention from the current token to the token immediately following the previous occurrence of the same token
- This pattern is qualitatively different from induction-free heads — it only appears on repeated sequences, not on novel text

**Ablation:**
- Ablating the identified induction heads sharply reduces the model's ability to copy the repeated continuation (next-token accuracy on the repeated span drops significantly)
- Heads outside the identified set show minimal effect on this specific task

**Direct Logit Attribution:**
- The induction heads show strong positive attribution toward the correct (repeated) next token, concentrated at the second occurrence of the pattern

**Logit Lens:**
- The correct next-token prediction should not be confidently present in early layers — it should sharpen specifically at the layer(s) where the induction heads write to the residual stream

## Pass/Fail criteria
- ✅ PASS: A small number of heads (not spread across all layers) show prefix-matching attention specifically on repeated sequences
- ✅ PASS: Ablating those heads measurably reduces copy accuracy on the repeated span
- ✅ PASS: DLA shows those same heads contributing positively to the correct next token
- ❌ FAIL: No heads show prefix-matching attention, or the effect is uniformly spread across all heads
- ❌ FAIL: Ablation of the identified heads has no measurable effect on copy accuracy
