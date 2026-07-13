# Seesaw v0 — Validation Ground Truth

Reference findings from published papers. Lens output should be compared against these.
All tasks run on GPT-2 Small unless noted. Each task has its own file — this is the index.

---

## Tasks

| # | Task | Paper | Model | File |
|---|---|---|---|---|
| 1 | Indirect Object Identification (IOI) | Wang et al. (2022) | GPT-2 Small | [01_ioi.md](01_ioi.md) |
| 2 | Greater-Than Circuit | Conmy et al. (2023) — ACDC | GPT-2 Small | [02_greater_than.md](02_greater_than.md) |
| 3 | Gender Bias in Occupation Pronouns | Vig et al. (2020) | GPT-2 Small/Medium | [03_gender_bias.md](03_gender_bias.md) |
| 4 | Factual Association (ROME) | Meng et al. (2022) | GPT-2 XL | [04_factual_recall_rome.md](04_factual_recall_rome.md) |
| 5 | Induction Heads | Olsson et al. (2022) | GPT-2 Small | [05_induction_heads.md](05_induction_heads.md) |
| 6 | Successor Heads | Gould et al. (2023) | GPT-2 Small | [06_successor_heads.md](06_successor_heads.md) |
| 7 | Copy Suppression | McDougall et al. (2023) | GPT-2 Small | [07_copy_suppression.md](07_copy_suppression.md) |
| 8 | Docstring Circuit | Heimersheim & Janiak (2023) | attention-only-4l | [08_docstring_circuit.md](08_docstring_circuit.md) |

---

## Summary Table

| Task | Model | Key layers | Dominant component | Metric |
|---|---|---|---|---|
| IOI | GPT-2 Small | 7–11 | Attention heads | Logit diff IO−S |
| Greater-than | GPT-2 Small | 4–8 | Attention heads | Valid year probability |
| Gender bias | GPT-2 Small | 7–9 | Attention heads | Logit diff she−he |
| Factual recall | GPT-2 XL | 13–17 | MLP layers | Token rank / probability |
| Induction heads | GPT-2 Small | ~5–6 | Attention heads | Copy accuracy / prefix-match score |
| Successor heads | GPT-2 Small | mid-late | Attention heads | Successor-token logit, cross-type |
| Copy suppression | GPT-2 Small | ~10 | Single suppressive head | Naive-copy logit (should decrease when head present) |
| Docstring circuit | attention-only-4l | all 4 | Attention heads (full circuit) | Correct parameter-name logit |

**Note**: Tasks 1–3 are structurally similar (attention-head circuits, circuit_analysis mode).
Task 4 (factual recall) is deliberately different — it validates that Seesaw correctly identifies MLP-dominant mechanisms
and doesn't over-attribute everything to attention.

Tasks 5–8 add coverage that 1–4 don't: induction heads and successor heads test whether Lens can find *general-purpose*
circuits that recur across many prompt types (not just one task's specific vocabulary), copy suppression tests
whether Lens correctly identifies a *single* suppressive head rather than a distributed circuit, and the docstring
circuit tests whether Lens's tools work on a non-GPT-2 model without hardcoded assumptions about layer count.
