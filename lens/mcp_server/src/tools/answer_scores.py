"""Agent-facing tool: how strongly the model gives the answer on each prompt, before anything is changed.

No intervention and no library function behind it: one forward pass, read
at each prompt's answer position. It's the "is there a behaviour at all?"
check that every other tool's numbers are relative to.
"""

from pathlib import Path

import torch
from transformer_lens import HookedTransformer

from seesaw.interp._helpers import tokens_to_ids
from seesaw.interp.metrics import last_positions

from ..config import PLOTS_DIR
from ..models.schemas import ExperimentResult


def run_answer_scores(
    model: HookedTransformer,
    prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str] | None = None,
    pos: int | None = None,
    output_dir: Path = PLOTS_DIR,
) -> ExperimentResult:
    """The answer token's probability on each prompt, and its logit diff over `negative_tokens` if given.

    `pos` is the position read: by default each prompt's own last token; -1
    reads the batch's last position, which is padding for shorter prompts.
    Answers: "Does the model do this task at all, and how confidently?"
    """
    tokens = model.to_tokens(prompts)
    rows = torch.arange(len(prompts))
    at = last_positions(model, prompts) if pos is None else torch.full((len(prompts),), pos)
    pos_ids = torch.tensor(tokens_to_ids(model, positive_tokens))
    with torch.no_grad():
        final = model(tokens)[rows.to(tokens.device), at.to(tokens.device)].float().cpu()
    probs = final.softmax(-1)[rows, pos_ids]
    per_prompt = [{"prompt": p, "probability": round(probs[i].item(), 6)} for i, p in enumerate(prompts)]
    data = {"mean_probability": round(probs.mean().item(), 6), "per_prompt": per_prompt}
    if negative_tokens:
        lds = final[rows, pos_ids] - final[rows, torch.tensor(tokens_to_ids(model, negative_tokens))]
        for row, ld in zip(per_prompt, lds.tolist()):
            row["logit_diff"] = round(ld, 4)
        data["mean_logit_diff"] = round(lds.mean().item(), 4)
    return ExperimentResult(
        name="Answer Scores", tool="answer_scores", model_name=model.cfg.model_name,
        prompts=prompts, data=data, status="success",
    )
