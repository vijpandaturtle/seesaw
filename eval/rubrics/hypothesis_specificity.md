# Rubric: Hypothesis specificity & falsifiability

Grade the **specificity and falsifiability** of the hypotheses in a mechanistic
interpretability research plan.

A hypothesis is *specific* if it names a concrete, falsifiable prediction — a
particular head/layer/direction and a predicted effect (e.g. "head L9H6 will show
positive direct-logit attribution toward the indirect-object token"). It is *vague*
if it only gestures at a region or phenomenon ("attention heads in the middle
layers are probably involved").

## Scale (1–5)

- **5** — Every hypothesis names a concrete component and a predicted, measurable effect.
- **4** — Most hypotheses are concrete; one or two are loosely specified.
- **3** — Mixed: some testable predictions, some vague direction-only claims.
- **2** — Mostly vague; predictions are hard to falsify as written.
- **1** — Hypotheses are unfalsifiable or merely restate the question.

If the plan is too garbled or off-topic to grade meaningfully, return
`unable_to_grade` rather than guessing a score.

## Output

- `score`: integer 1–5
- `reasoning`: one or two sentences citing the specific hypotheses that drove the score
