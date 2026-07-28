"""Induction heads: the attention signature should be present and sharp.

Induction heads (Olsson et al., 2022) attend from the current token back to the
token that followed its previous occurrence, so on a repeated sequence they
concentrate attention on one earlier position rather than spreading it. They
appear in GPT-2 small from layer 5 onwards.

This checks the signature the attention sweep can actually show: that some head
attends sharply somewhere, rather than every head being diffuse.
"""

from conftest import require_tool

# A head attending >30% of its mass to a single position is doing something
# specific; diffuse heads on a short sequence sit far below this.
SHARP_ATTENTION = 0.30


def _max_attention_per_layer(result: dict) -> dict[int, float]:
    """layer -> the sharpest single-position attention any head achieved."""
    out = {}
    for key, value in result.get("data", {}).items():
        if not key.startswith("layer_"):
            continue
        per_head = value.get("max_attn_per_head") or []
        if per_head:
            out[int(key.removeprefix("layer_"))] = max(per_head)
    return out


def test_attention_pattern_ran(by_tool):
    require_tool(by_tool, "attention_pattern")


def test_some_head_attends_sharply(by_tool):
    """At least one head somewhere concentrates attention. If nothing does, the
    sequence probably has no repetition for induction to latch onto."""
    best = 0.0
    for result in require_tool(by_tool, "attention_pattern"):
        per_layer = _max_attention_per_layer(result)
        if per_layer:
            best = max(best, max(per_layer.values()))
    assert best >= SHARP_ATTENTION, (
        f"sharpest attention anywhere is {best:.3f}; no head is concentrating "
        f"on a single position, so there's no induction signature to find"
    )


def test_layers_were_actually_swept(by_tool):
    """A single-layer sweep can't tell you where induction heads live."""
    for result in require_tool(by_tool, "attention_pattern"):
        layers = result.get("data", {}).get("layers_analysed") or []
        assert len(layers) >= 2, (
            f"attention pattern swept only {layers}; induction heads emerge in "
            f"middle layers, so a single-layer sweep can't locate them"
        )
