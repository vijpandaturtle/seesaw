"""Feature Extraction — Superposition & feature extraction. Requires pretrained SAE weights (e.g. via Neuronpedia) as a new dependency, not just a new hook.

Built:
- sae — load pretrained SAEs via SAELens; read their hook point and Neuronpedia id
- neuronpedia — API client: feature explanations, server-side feature search
- feature_attribution — activation × gradient score for every SAE feature
- feature_search — which features fire on prompts, computed by Neuronpedia (no local model)

Planned: see `lens/mcp_server/src/tools/feature_extraction/README.md`.
"""
