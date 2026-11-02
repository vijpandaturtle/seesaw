# Seesaw use-cases: what to ask, and which pathway answers it

A pathway is a fixed, hand-coded investigation with a verdict at the end
([`seesaw/pathways/`](../seesaw/pathways/)). This guide sorts them by what you want to find out, and
gives sample questions for each one. Read it to pick a pathway and to see what a good question for
it looks like.

## Pick a purpose

| Purpose | The question you're asking | Pathway |
|---|---|---|
| [Understand a capability](#understand-a-capability) | How does the model do X? | `circuit_discovery` |
| [Audit a reliance](#audit-a-reliance) | Does the model use attribute X, and does removing it remove the behaviour? | `attribute_audit` |
| [Explain an output](#explain-an-output) | Why did it say X here? Which features drive it? | `feature_explain` (gpt2 only) |
| [Compare models](#reproducing-a-result) | Does model B do X the same way as model A? | any of the above, rerun on the other model |
| [Not runnable yet](#not-runnable-yet) | Debug a failure, replicate a claim, steer | none yet |

Every pathway scores one answer token against one distractor token per prompt. A good question
therefore has an answer the model gives as **a single next token**: `" Mary"` vs `" John"`,
`" she"` vs `" he"`. If you can't name the correct token and the wrong one, narrow the question
until you can.

★ marks a question whose answer is already known on GPT-2 small, checked in
[`tests/test_pathways_gpt2.py`](../tests/test_pathways_gpt2.py) or established in the literature.
Start with these: when the verdict comes back you know what it should have said.

## Running one

A pathway always runs as part of a research job. You ask the question and name the pathway, and:

1. **Scout** researches the question and chooses the pathway's inputs: prompts, answer tokens,
   contrasts. The worker checks them (aligned lists, every answer a single token for the model) and
   sends Scout any problems to fix, up to twice.
2. **You approve** the plan and the inputs at the gate, before anything runs on a model.
3. **Lens** runs the pathway first, on exactly those inputs, then any extra experiments in the
   plan.
4. **Quill** critiques the result, with the pathway's verdict as the primary evidence.

From the dashboard, type `/` in the composer and pick the pathway. From a terminal:

```bash
seesaw jobs submit "How does GPT-2 complete 'The war lasted from 1732 to 17' with a later year?" --pathway circuit_discovery --follow
```

The inputs each pathway takes are listed in [`seesaw/pathways/catalog.py`](../seesaw/pathways/catalog.py).

### Reproducing a result

Scout uses an LLM and searches the web, so asking the same question twice gives different inputs, and
there's no seed that fixes that. The pathways themselves are deterministic (their random baselines are
seeded): the same pathway version on the same model and inputs gives the same verdict.

So a result is reproduced by **rerunning** its job: a new job starts at Lens with the same plan and
the inputs approved last time, and Scout doesn't run again. Use the **Rerun** button on the job page,
or:

```bash
seesaw jobs rerun <id> --follow
```

Add `--model` (or pick another model next to the button) to run the same experiment on another model:
that's how to compare models on identical prompts. The inputs are checked against the new model first,
so an answer that's two tokens in its tokenizer is refused before anything runs. Lens's extra
experiments are read from the plan by an LLM again, so only the pathway is guaranteed to repeat.

For fixed inputs with no job at all (the ★ checks below, or trying a pathway in a notebook), call the
pathway directly: `seesaw.pathways.circuit_discovery.run(model, prompts, positive_tokens,
negative_tokens, output_dir)`. That's what the known-answer tests do.

## Understand a capability

**Pathway:** `circuit_discovery`. It nominates the heads that matter, verifies them by patching, and
checks that the circuit on its own reproduces the behaviour.

**Inputs:** `prompts`, `positive_tokens`, `negative_tokens`. Optionally `corrupted_prompts` (the same
prompt with the key information changed, same token length) and `reference_prompts` (a neutral
distribution used as ablation means).

- ★ How does GPT-2 work out who received the object in "When Mary and John went to the store, John
  gave a drink to"? *(indirect object identification. Expected: name-mover, S-inhibition and
  negative name-mover heads.)*
- How does GPT-2 complete "The war lasted from 1732 to 17" with a year later than the start?
  *(greater-than, Hanna et al. 2023.)*
- How does GPT-2 continue a sequence of random tokens it has already seen once? *(induction heads.)*
- How does GPT-2 choose "are" rather than "is" after a plural subject?
- How does GPT-2 close a bracket with the right type, `)` or `]`?

## Audit a reliance

**Pathway:** `attribute_audit`. It tests whether the model's output depends on an attribute, finds the
attribute's direction, and checks whether removing that direction removes the behaviour without
damaging the model.

**Inputs:** `prompts`, `counterfactual_prompts` (the same prompt with the attribute swapped),
`positive_tokens`, `negative_tokens`. Optionally `control_prompts` (attribute-free prompts showing the
baseline) and `control_texts` (ordinary text for checking the edit is harmless).

- ★ Does GPT-2 use the gender a job title implies to choose "she" or "he"? Does removing that
  direction remove the bias? *(e.g. "The nurse said that" vs "The engineer said that", `" she"` vs
  `" he"`. Expected: bias found and removed.)*
- ★ Control: does GPT-2 use an irrelevant attribute, like the food someone ate ("The man who ate the
  apple said that" vs "... the pizza said that"), to choose the pronoun? *(expected: not flagged. Run this next to the one above: an audit that flags everything
  is worthless.)*
- Does GPT-2 use the nationality a name implies when it guesses where someone lives ("Hiroshi lives
  in")?
- Does GPT-2 change a review's sentiment when only the reviewer's name changes?
- Does GPT-2 take "not" into account when it chooses a sentiment word, or ignore the negation?

## Explain an output

**Pathway:** `feature_explain`. It finds the sparse autoencoder (SAE) features that drive the answer
and describes what each one responds to. It only runs on `gpt2`, because it uses a GPT-2 small SAE
(`gpt2-small-res-jb`, layer 8 residual stream).

**Inputs:** `prompts`, `positive_tokens`, `negative_tokens`.

- ★ Which SAE features make GPT-2 answer "Mary" rather than "John" in the IOI prompt?
- Why does GPT-2 say "Paris" after "The Eiffel Tower is in"?
- Which features fire when GPT-2 chooses "he" after "The doctor said that"?
- What drives GPT-2 to close a quotation mark at the end of a quoted sentence?

## Not runnable yet

These purposes have no pathway yet. You can still ask them as plain research jobs
(`seesaw jobs submit "<question>"`), where Scout plans the experiments freely and the result has no
fixed verdict.

| Purpose | Sample question | What a pathway would need |
|---|---|---|
| Debug a failure | Why does GPT-2 get IOI wrong when the two names are the same gender and length? | Run correct and failing prompts through the same circuit and compare |
| Replicate a claim | Does the IOI circuit from Wang et al. (2022) show up in Pythia-160m? | `circuit_discovery` with inputs taken from the paper, checked against its heads |
| Trace a multi-hop fact | How does the model get from "Dallas" to "Austin"? (Ameisen et al. 2025 found a route through "Texas" in Gemma-2-2B) | Attribution graphs, which Lens's `circuit_tracing` tool reads from Neuronpedia but no pathway uses yet |
| Steer or edit | Can I remove the gender direction without hurting GPT-2 on ordinary text? | The ablation step inside `attribute_audit`, exposed on its own |
