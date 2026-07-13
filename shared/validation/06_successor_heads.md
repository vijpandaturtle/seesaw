# 6. Successor Heads
**Paper**: Gould et al. (2023) — *Successor Heads: Recurring, Interpretable Attention Heads In The Wild*
**Model**: GPT-2 Small
**Mode**: `circuit_analysis`
**Input**: Ordinal/sequential prompts, e.g. `"Monday, Tuesday,"` (expects `" Wednesday"`) or `"one, two,"` (expects `" three"`)
**Metric**: Logit of the correct successor token vs. other plausible completions

## What Lens should find

**Direct Logit Attribution:**
- A small set of heads contribute strong positive attribution toward the correct successor token across multiple ordinal sequence types (days, months, numbers) — the same heads should show up across different sequence *types*, not just one

**Ablation:**
- Ablating the identified successor heads reduces the logit for the correct successor token across all tested sequence types (days, months, numbers), not just one
- This cross-type consistency is the key finding of the paper — a head that only works for days-of-the-week is not a "successor head" in their sense

**Attention Patterns:**
- Successor heads attend to the current token itself (not to a distant context token) — the "successor" computation is closer to a lookup/increment operation on the current token than a retrieval from earlier context

## Pass/Fail criteria
- ✅ PASS: The same 1-3 heads show positive DLA across at least two different ordinal sequence types (e.g. both days and numbers)
- ✅ PASS: Ablating those heads reduces successor-token logit across all tested sequence types
- ❌ FAIL: Different heads are responsible for each sequence type with no overlap
- ❌ FAIL: No heads show consistent cross-type successor behaviour
