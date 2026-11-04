# Seesaw use-cases: what to ask, and which pathway answers it

A pathway is a fixed, hand-coded investigation with a verdict at the end
([`seesaw/pathways/`](../seesaw/pathways/)). This guide sorts them by what you want to find out, and
gives sample questions for each one. Read it to pick a pathway and to see what a good question for
it looks like.

## Pick a purpose

| Purpose | The question you're asking | Pathway |
|---|---|---|
| [Understand a capability](#understand-a-capability) | How does the model do X? | `circuit_discovery` |
| [Find and steer a concept](#find-and-steer-a-concept) | Is concept X a direction the model uses? Can I remove or steer it safely? | `concept_direction` (`attribute_audit` is its older name) |
| [Build a detector](#build-a-detector) | Would a probe on the activations flag X on new texts, better than the words alone? | `probe_monitor` |
| [Audit a spurious dependence](#audit-a-spurious-dependence) | Does the answer depend on something it shouldn't? Can that be erased without breaking the task? | `spurious_audit` |
| [Locate a fact](#locate-a-fact) | Where does the model retrieve a fact? | `factual_recall` |
| [Edit a fact](#edit-a-fact) | Is the layer where a fact is traced the best layer to edit it? | `editing_comparison` |
| [Explain an output](#explain-an-output) | Why did it say X here? Which features drive it? | `feature_explain` (gpt2 only) |
| [Diff two models](#diff-two-models) | What did a fine-tune change, and in which weights? | `model_diffing` |
| [Compare models](#reproducing-a-result) | Does model B do X the same way as model A? | any of the above, rerun on the other model |
| [Not runnable yet](#not-runnable-yet) | Debug a failure, replicate a claim, attribute to the context | none yet |

Most pathways score one answer token against one distractor token per prompt. A good question for
them has an answer the model gives as **a single next token**: `" Mary"` vs `" John"`,
`" she"` vs `" he"`. If you can't name the correct token and the wrong one, narrow the question
until you can. `probe_monitor` and `model_diffing` take texts, not answer tokens.

Every step of every pathway is labelled with how strong its evidence is (Correlational, Estimate
or Causal) and whether its controls ran; see [`shared/evidence.py`](../shared/evidence.py).

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
so an answer that's two tokens in its tokenizer is refused before anything runs.

Lens's extra experiments repeat too: a rerun replays the ones the source job ran, with the same tools,
prompts and arguments, instead of having an LLM read the plan again, and adds no new follow-ups. What
can differ is only what an LLM writes about the results: Lens's summaries and Quill's critique. On
another model, an experiment that names something the model doesn't have (a layer past its last one,
say) fails and is reported as failed. If the source job's results aren't on the machine running the
rerun, Lens reads the plan again, and the Timeline shows that it did.

There's no sampling setting to make the LLM parts repeat: the API no longer accepts `temperature`,
and even `temperature` 0 never guaranteed identical output. Reproducibility comes from recording what
was decided and replaying it.

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

## Find and steer a concept

**Pathway:** `concept_direction`. It tests whether the output depends on a concept, finds the
concept's direction on half the prompt pairs, and on the other half checks whether removing it
removes the behaviour and adding it steers the behaviour, both against random directions and with a
loss check. Last, it folds the removal into the weights and checks the result matches, so the edit
can ship as a checkpoint. `attribute_audit` runs the same pathway; version 1 of it fitted and tested
the direction on the same pairs, which let it pass by construction.

**Inputs:** `prompts`, `counterfactual_prompts` (the same prompt with the concept swapped),
`positive_tokens`, `negative_tokens`, at least 6 pairs. Optionally `control_prompts` (concept-free
prompts showing the baseline) and `control_texts` (ordinary text for checking the edit is harmless).

- ★ Does GPT-2 use the gender a job title implies to choose "she" or "he"? Does removing that
  direction remove the bias? *(e.g. "The nurse said that" vs "The engineer said that", `" she"` vs
  `" he"`. Expected: `partly_uses_direction`. On held-out pairs, removal closes about a fifth of the
  gap, well beyond random directions, and steering moves it all the way.)*
- ★ Control: does GPT-2 use an irrelevant attribute, like the food someone ate ("The man who ate the
  apple said that" vs "... the pizza said that"), to choose the pronoun? *(expected: not flagged. Run this next to the one above: an audit that flags everything
  is worthless.)*
- Does GPT-2 use the nationality a name implies when it guesses where someone lives ("Hiroshi lives
  in")?
- Does GPT-2 change a review's sentiment when only the reviewer's name changes?
- Does GPT-2 take "not" into account when it chooses a sentiment word, or ignore the negation?

## Build a detector

**Pathway:** `probe_monitor`. It trains a probe at every layer on your texts, picks the layer on
those alone, then tests it on held-out texts against shuffled labels, a probe on the embeddings
alone, and a bag-of-words classifier with no model. The question is the deployment one: is a probe
on the activations worth it over the cheapest classifier?

**Inputs:** `prompts` and `contrast_prompts` (training texts with and without the property),
`test_prompts` and `test_contrast_prompts` (held-out texts, ideally from another source or topic).

- ★ Can a GPT-2 probe trained on movie reviews detect the sentiment of product reviews? *(expected:
  `lexical_only`. It transfers, but the sentiment words alone do as well.)*
- Can a probe tell a question the model will answer wrongly from one it will answer correctly?
- Can a probe flag prompts that try to override earlier instructions?

## Audit a spurious dependence

**Pathway:** `spurious_audit`. For a task whose answer shouldn't depend on an attribute, it measures
how much swapping the attribute moves the answer, then erases the attribute with LEACE (fitted on
half the pairs, applied at every layer on the other half) and checks that the dependence goes and
the task stays, against random erasers.

**Inputs:** `prompts`, `counterfactual_prompts` (only the attribute swapped), `positive_tokens`,
`negative_tokens` (the task's right and wrong answers), at least 6 pairs. Optionally `control_texts`.

- ★ Does GPT-2's sentiment judgement depend on the reviewer's name? *(expected: `no_reliance`.)*
- ★ Does a gendered name pull GPT-2 off an occupation the context states ("My friend Mary has
  always worked as a pilot. Everyone knows that Mary is a", `" pilot"` vs `" secretary"`)?
  *(expected: reliance found, and erasure removes it far beyond random erasers.)*

## Locate a fact

**Pathway:** `factual_recall`. Causal tracing (Meng et al. 2022): it noises the subject's
embeddings, then restores the residual stream, MLP outputs or attention outputs at each layer and
position, and reports where the answer's probability comes back.

**Inputs:** `prompts`, `subjects` (the subject as written in each prompt), `positive_tokens`.

- ★ Where does GPT-2 recall that the Eiffel Tower is in Paris? *(expected: the residual stream
  carries the fact at the subject's last token early and at the final token late. On GPT-2 small
  the MLP effect at the subject is weak outside layers 0 and 1, so the verdict is `late_site`, not the
  ROME pattern Meng et al. found in GPT-2 XL.)*

## Edit a fact

**Pathway:** `editing_comparison`. It traces each fact, then makes a ROME edit at every layer and
scores each on efficacy, paraphrase and specificity, to test whether the traced layer is the best
one to edit (Hase et al. 2023 found it usually isn't).

**Inputs:** `prompts`, `subjects`, `positive_tokens`, `target_tokens` (the new answers),
`paraphrase_prompts`, `neighborhood_prompts` and `neighborhood_tokens` (a different subject with the
same relation, and its answer).

- ★ Can GPT-2 be taught that Michael Jordan played soccer, and where is the best layer to do it?
  *(expected: edits work mid-model and can't work at the last layer.)*

## Diff two models

**Pathway:** `model_diffing`. It compares the job's model with `other_model` on your prompts: the KL
divergence per prompt, what each predicts where they differ most, the tokens the change made more
likely, and (same architecture only) how much of the difference each component's weights account
for, by swapping them in one at a time.

**Inputs:** `prompts`, and `other_model` (a single TransformerLens model name, not a list).

- ★ Planted: a copy of GPT-2 with noise added to block 5's MLP. *(expected: `localized` at block 5.)*
- What did instruction tuning change in a Pythia checkpoint, and is it in a few blocks or everywhere?

## Explain an output

**Pathway:** `feature_explain`. It finds the sparse autoencoder (SAE) features that drive the answer
and describes what each one responds to. It sweeps GPT-2 small SAEs (`gpt2-small-res-jb`, residual
stream at layers 4, 6, 8 and 10) and explains the layer where switching features off accounts for
the most, so it only runs on `gpt2`.

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
| Attribute to the context | Which retrieved passage made the model give this answer? | Input and context attribution (gradient × input, ablating one passage at a time) |
