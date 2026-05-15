import torch
from transformer_lens import HookedTransformer


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
