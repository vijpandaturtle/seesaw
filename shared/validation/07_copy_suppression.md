# 7. Copy Suppression
**Paper**: McDougall et al. (2023) — *Copy Suppression: Comprehensively Understanding an Attention Head*
**Model**: GPT-2 Small
**Mode**: `circuit_analysis`
**Input**: Prompts where an early token is a plausible (but incorrect) continuation, e.g. `"All's fair in love and war. The saying 'love and'"` — a naive copy would predict `" love"` again, but the correct continuation is different
**Metric**: Logit of the naively-copied (incorrect) token vs. the correct token, before/after ablating the suppression head

## What Lens should find

**Direct Logit Attribution:**
- One specific head — **L10H7** in GPT-2 Small — shows strong **negative** attribution toward tokens that appeared earlier in the context and would be a naive copy. The paper shows this copy-suppression mechanism explains **76.9%** of L10H7's total impact on the model's output, i.e. it is nearly the whole story for that head, not one factor among many

**Ablation:**
- Ablating this head *increases* the logit of the naively-copied token — i.e. removing it makes the model more repetitive, not less. This is the same "suppressive head" signature seen in the IOI paper's S-Inhibition heads and Seesaw's own gender-bias run (Tool 5 found L11H08 and L09H05 acting the same way)

**Attention Patterns:**
- The head attends from the current position back to the earlier occurrence of the token it is about to suppress

## Pass/Fail criteria
- ✅ PASS: L10H7 (or the equivalent head Lens identifies) shows strong negative DLA on the naively-copied token
- ✅ PASS: The copy-suppression mechanism accounts for a large majority (paper reports 76.9%) of that head's total attribution — not just a minor contributing factor
- ✅ PASS: Ablating that head increases (not decreases) the naive-copy logit
- ❌ FAIL: No head shows a clear negative-attribution, suppressive signature
- ❌ FAIL: Ablating the candidate head decreases the naive-copy logit (wrong direction — would mean it's a promoter, not a suppressor)
