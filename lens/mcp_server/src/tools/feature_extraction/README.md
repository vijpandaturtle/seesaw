# Feature Extraction

Superposition & feature extraction. Requires pretrained SAE weights (e.g. via Neuronpedia) as a new dependency, not just a new hook.

## Built in `seesaw.interp.feature_extraction`
The computation lives in the library; the Lens wrapper is in `lens/mcp_server/src/tools/` and the tool is in `shared/lens_tools.py`.
- `sae` — load pretrained SAEs through SAELens (no training); read hook point and Neuronpedia id
- `neuronpedia` — API client: feature explanations and dashboards (covers feature_dashboards), server-side feature search; responses cached on disk
- `feature_attribution` — activation × gradient score for every SAE feature, labelled from Neuronpedia
- `feature_search` — which features fire on prompts, computed by Neuronpedia with no local model

## Not yet implemented
- **scaling_monosemanticity** — Techniques for keeping SAE features monosemantic as model/dictionary size scales up.
- **sae_variants** — Gated SAEs, TopK SAEs, and other architectural variants beyond the vanilla SAE.
- **temporal_representations** — Feature extraction that accounts for how representations evolve across a sequence, not just a single position.
- **transcoders** — SAE-like decomposition applied across a whole MLP/attention block rather than a single activation.
- **crosscoders** — Feature extraction shared across multiple models or checkpoints simultaneously.
