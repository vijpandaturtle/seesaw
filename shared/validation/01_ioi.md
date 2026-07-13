# 1. Indirect Object Identification (IOI)
**Paper**: Wang et al. (2022) — *Interpretability in the Wild*
**Model**: GPT-2 Small
**Mode**: `circuit_analysis`
**Input**: `"When Mary and John went to the store, John gave a drink to"`
**Metric**: Logit diff = logit(IO token) − logit(Subject token)

## What Lens should find

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

## Pass/Fail criteria
- ✅ PASS: L9H6 and L9H9 appear in top-5 DLA heads with positive contribution
- ✅ PASS: Ablating L9H6 + L9H9 reduces logit diff by > 30%
- ✅ PASS: Activation patching recovery peaks in layers 7–11 at subject/IO positions
- ❌ FAIL: If top DLA heads are in layers 0–5
- ❌ FAIL: If ablation effects are uniform across all heads (no clear circuit)
