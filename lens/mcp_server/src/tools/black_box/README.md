# Black Box

No model internals required at all — input/output only. Cheapest tool here, and the only one that survives a closed model or a backend change.

## Built in `seesaw.interp.black_box`
The computation lives in the library; the Lens wrapper is in `lens/mcp_server/src/tools/` and the tool is in `shared/lens_tools.py`.
- `counterfactual` — swap one attribute in the input, measure how far the output moves (covers counterfactual_resampling for paired inputs)

## Not yet implemented
