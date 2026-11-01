# Hidden State Decoding

Turning a raw hidden-state vector into a natural-language label, using the model's own generation ability.

## Not yet implemented
- **patchscopes** — Patch a hidden state into a diagnostic target prompt and read the decoded interpretation. Scoped since Part 2 of the build log — next up.
- **selfie** — Self-interpretation via a model explaining its own internal representations in its own words.
- **concept_injection** — Inject a known concept vector and observe how the model's generation changes.
- **latentqa** — Question-answering interface directly over a model's latent representations.
- **activation_oracles** — A trained oracle model that answers questions about another model's activations.
- **natural_language_autoencoders** — Autoencode a hidden state through natural language as the bottleneck, not a numeric latent.
