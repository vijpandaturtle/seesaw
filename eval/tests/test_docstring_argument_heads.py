"""Docstring circuit: prediction should depend on earlier argument names.

The docstring task (Heimersheim & Janiak, 2023) has GPT-2 small predict the
next parameter name in a docstring, which requires attending back to where that
argument was defined in the signature. The circuit involves induction-like and
positional heads in the middle layers.

The observable here is attention structure: predicting the next argument means
attending to specific earlier tokens, not to the sequence uniformly.
"""

from conftest import require_tool

FOCUSED_ATTENTION = 0.25


def _final_token_rows(result: dict):
    """Per-head attention from the final token, for each swept layer."""
    for key, value in result.get("data", {}).items():
        if key.startswith("layer_"):
            for row in value.get("final_token_attn") or []:
                if row:
                    yield int(key.removeprefix("layer_")), row


def test_attention_pattern_ran(by_tool):
    require_tool(by_tool, "attention_pattern")


def test_prompts_are_long_enough_to_reference_back(by_tool):
    """Attending back to an argument definition needs tokens to attend back
    to. A short prompt can't exhibit the behaviour."""
    for result in require_tool(by_tool, "attention_pattern"):
        tokens = result.get("data", {}).get("token_strs") or []
        assert len(tokens) >= 8, (
            f"prompt is {len(tokens)} tokens; too short to contain a signature "
            f"and a docstring for the model to reference back to"
        )


def test_some_head_focuses_on_an_earlier_token(by_tool):
    """A head that spreads attention evenly isn't doing argument lookup."""
    best = 0.0
    for result in require_tool(by_tool, "attention_pattern"):
        for _layer, row in _final_token_rows(result):
            best = max(best, max(row))
    assert best >= FOCUSED_ATTENTION, (
        f"the most focused head puts only {best:.3f} of its attention on any "
        f"single earlier token; nothing is doing argument lookup"
    )


def test_attention_is_not_only_on_the_first_token(by_tool):
    """Heads parked on the BOS token are resting, not computing. If every head
    peaks at position 0, no lookup is happening."""
    for result in require_tool(by_tool, "attention_pattern"):
        rows = list(_final_token_rows(result))
        if not rows:
            continue
        peaks_beyond_bos = [
            row for _layer, row in rows if row.index(max(row)) != 0
        ]
        assert peaks_beyond_bos, (
            "every head's attention peaks on the first token; that's the "
            "standard resting position, so no head is referencing the signature"
        )
