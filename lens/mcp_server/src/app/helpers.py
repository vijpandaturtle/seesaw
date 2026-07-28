import torch
from transformer_lens import HookedTransformer

# Rows per forward pass in a batched sweep. Sweeps replicate the prompts once
# per intervention, so this bounds activation memory: raising it trades memory
# for speed, and it's the knob to turn down on a large model.
MAX_SWEEP_ROWS = 64


def sweep_chunks(n_interventions: int, n_prompts: int, max_rows: int = MAX_SWEEP_ROWS):
    """Split interventions into groups small enough to batch in one pass.

    Yields (start, stop) index pairs covering range(n_interventions). Each
    group becomes a batch of len(group) * n_prompts sequences.
    """
    per_chunk = max(1, max_rows // max(1, n_prompts))
    for start in range(0, n_interventions, per_chunk):
        yield start, min(start + per_chunk, n_interventions)


def row_logit_diffs(
    logits: torch.Tensor,
    io_token_ids: list[int],
    subject_token_ids: list[int],
    n_prompts: int,
    pos: int = -1,
) -> torch.Tensor:
    """Per-row logit diff for a batch built by tiling prompts.

    Rows are laid out intervention-major — `tokens.repeat(k, 1)` — so row r
    holds prompt `r % n_prompts` under intervention `r // n_prompts`. That
    layout lets the caller reshape the result to [n_interventions, n_prompts]
    and average over prompts, matching what get_logit_diff returns per pass.

    Args:
        logits: [rows, seq, d_vocab] from one batched forward pass.
        io_token_ids: Correct token id per prompt.
        subject_token_ids: Incorrect token id per prompt.
        n_prompts: Number of distinct prompts tiled into the batch.
        pos: Token position to read logits from (-1 = last token).

    Returns:
        [rows] tensor of (IO logit - subject logit).
    """
    rows = torch.arange(logits.shape[0], device=logits.device)
    prompt_of_row = rows % n_prompts
    io = torch.tensor(io_token_ids, device=logits.device)[prompt_of_row]
    s = torch.tensor(subject_token_ids, device=logits.device)[prompt_of_row]
    return logits[rows, pos, io] - logits[rows, pos, s]


def get_logit_diff(
    model: HookedTransformer,
    tokens: torch.Tensor,
    io_token_ids: list[int],
    subject_token_ids: list[int],
    pos: int = -1,
) -> float:
    """Run the model and return the mean logit diff (IO logit - subject logit) at position `pos`.

    This is the standard IOI metric from Wang et al. (2022).

    Args:
        model: Loaded HookedTransformer.
        tokens: Tokenised prompts tensor [batch, seq].
        io_token_ids: Correct (indirect object) token IDs, one per prompt.
        subject_token_ids: Incorrect (subject) token IDs, one per prompt.
        pos: Token position to read logits from (-1 = last token).

    Returns:
        Mean logit difference across the batch (scalar float).
    """
    with torch.no_grad():
        logits = model(tokens)   # [batch, seq, d_vocab]
    diffs = []
    for i, (io_id, s_id) in enumerate(zip(io_token_ids, subject_token_ids)):
        diff = logits[i, pos, io_id] - logits[i, pos, s_id]
        diffs.append(diff.item())
    return sum(diffs) / len(diffs)


def tokens_to_ids(model: HookedTransformer, token_strs: list[str]) -> list[int]:
    """Convert a list of token strings (with leading space) to vocab IDs.

    Args:
        model: Loaded HookedTransformer.
        token_strs: Token strings, e.g. [" Mary", " John"].

    Returns:
        List of integer vocab IDs.
    """
    return [model.to_single_token(t) for t in token_strs]
