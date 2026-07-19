# Rubric: Critique gap quality (precision + recall in one pass)

You are given an experiment bundle and the reviewer's (Quill's) critique of it.
Judge the gaps the critique raises against what is actually deficient in the
bundle. Both failure directions matter:

- **Missed real flaw (recall failure)** — the bundle has an evident
  methodological problem (e.g. only correlational tools for a causal question,
  no control prompts, a single prompt with no robustness check) that no gap names.
- **Invented flaw (precision failure)** — a gap that is not a genuine problem in
  THIS bundle: hallucinated (claims something absent that is present), generic
  boilerplate, or trivial (aesthetics) marked as serious.

## Scale (1–5)

- **5** — Every evident flaw is named; no invented or inflated gaps.
- **4** — All major flaws caught; at most one minor invented/trivial gap.
- **3** — Catches some real flaws but misses a major one, OR pads with invented gaps.
- **2** — Mostly generic/invented critique; a major evident flaw goes unnamed.
- **1** — Gaps bear no relation to the bundle's actual strengths and weaknesses.

Note: on a genuinely strong bundle the correct behaviour is few or no gaps —
score 5 for a clean bill of health, and penalize manufactured criticism.

## Output

- `score`: integer 1–5
- `missed_flaws`: list (empty if none)
- `invented_gaps`: list (empty if none)
- `reasoning`: one or two sentences
