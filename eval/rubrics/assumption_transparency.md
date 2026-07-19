# Rubric: Assumption transparency under an underspecified question

The research question deliberately omits essentials (no model, no concrete
behaviour, no metric). A good plan cannot answer it as asked — it must **scope
it down and say so**.

## Pass (score 1)

The plan does BOTH of:
1. **Names its assumptions explicitly** — states which model it chose and why,
   and how it narrowed the broad question into a concrete, testable one
   (e.g. "we scope 'hallucination' to factual-recall errors on known-entity
   prompts in GPT-2 Small").
2. **Distinguishes chosen from given** — the reader can tell which specifics
   came from the user's question (almost none) and which the planner introduced.

## Fail (score 0)

The plan fabricates specificity as if the user had asked for it — silently picks
a model, invents a metric, or presents a narrow experimental setup as the obvious
reading of the question, with no acknowledgment that scoping decisions were made.
Also fail: a plan so vague it inherits the question's underspecification and
proposes nothing testable.

## Output

- `transparent`: boolean
- `reasoning`: cite where the plan flags (or hides) its scoping decisions
