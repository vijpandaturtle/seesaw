# Feature Extraction

Superposition & feature extraction. Requires pretrained SAE weights (e.g. via Neuronpedia) as a new dependency, not just a new hook.

## Not yet implemented
- **sparse_autoencoders** — Train or load an SAE to decompose polysemantic activations into monosemantic features.
- **feature_dashboards** — Generate a human-readable dashboard of what a given SAE feature activates on.
- **scaling_monosemanticity** — Techniques for keeping SAE features monosemantic as model/dictionary size scales up.
- **sae_variants** — Gated SAEs, TopK SAEs, and other architectural variants beyond the vanilla SAE.
- **temporal_representations** — Feature extraction that accounts for how representations evolve across a sequence, not just a single position.
- **transcoders** — SAE-like decomposition applied across a whole MLP/attention block rather than a single activation.
- **crosscoders** — Feature extraction shared across multiple models or checkpoints simultaneously.
