# Probing

Reading whether a specific concept or property is linearly (or non-linearly) decodable from activations.

## Built in `seesaw.interp.probing`
The computation lives in the library; the Lens wrapper is in `lens/mcp_server/src/tools/` and the tool is in `shared/lens_tools.py`.
- `linear_probe` — per-layer cross-validated logistic probe, plus the difference-of-means (CAA-style) direction at the best layer

## Not yet implemented
- **truthfulness_probing** — Probing classifiers specifically targeting truthful vs. untruthful internal states.
- **lat** — Linear Artificial Tomography — a specific linear-probing methodology.
- **attention_probes** — Probing classifiers applied to attention patterns/weights specifically rather than residual stream activations.
