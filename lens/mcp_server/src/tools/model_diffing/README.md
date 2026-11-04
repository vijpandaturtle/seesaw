# Model Diffing

Comparing two model checkpoints — needs two models as input, a different tool shape than everything else here.

## Built in `seesaw.interp.model_diffing`
- `diff` — per-prompt KL between two checkpoints, logit-diff amplification, the tokens the change favours, and (same architecture) the share of the KL each component's weights close when swapped in (library and pathways only; no Lens tool yet, since Lens loads one model)

## Not yet implemented
- **feature_level_diffing** — Comparing SAE features present in one checkpoint vs. another.
- **finetuning_traces** — Identifying which components were most affected by a finetuning run.
