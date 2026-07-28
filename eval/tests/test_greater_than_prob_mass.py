"""Greater-than: probability mass should sit on years after the start year.

Hanna et al. (2023) show GPT-2 small implements a greater-than operation: given
"The war lasted from 17X to 17", it puts most of its mass on two-digit
completions greater than X.

Logit lens reports the top tokens per layer, so the observable is that the
final-layer prediction is numeric and skews high, rather than the model
defaulting to some unrelated token.
"""

from conftest import require_tool


def _final_layer(result: dict) -> dict | None:
    layers = result.get("data", {}).get("layers") or []
    return layers[-1] if layers else None


def test_logit_lens_ran(by_tool):
    require_tool(by_tool, "logit_lens")


def test_layers_were_swept(by_tool):
    """The point of logit lens is watching the prediction form across depth."""
    for result in require_tool(by_tool, "logit_lens"):
        layers = result.get("data", {}).get("layers") or []
        assert len(layers) >= 4, (
            f"logit lens reported {len(layers)} layers; too few to see where "
            f"the greater-than computation resolves"
        )


def test_final_prediction_is_numeric(by_tool):
    """A non-numeric top token means the prompt never set up the task."""
    for result in require_tool(by_tool, "logit_lens"):
        final = _final_layer(result)
        if not final:
            continue
        tokens = final.get("top_tokens") or []
        numeric = [t for t in tokens if t.strip().isdigit()]
        assert numeric, (
            f"final-layer top tokens {tokens} contain no digits; the prompt "
            f"isn't eliciting a year completion, so there's no greater-than "
            f"behaviour to measure"
        )


def test_prediction_sharpens_with_depth(by_tool):
    """Early layers are diffuse and late layers commit. If the top probability
    never rises, the model isn't resolving the task across depth."""
    for result in require_tool(by_tool, "logit_lens"):
        layers = result.get("data", {}).get("layers") or []
        if len(layers) < 4:
            continue
        first = (layers[0].get("top_probs") or [0])[0]
        last = (layers[-1].get("top_probs") or [0])[0]
        assert last > first, (
            f"top-token probability went from {first:.3f} at layer 0 to "
            f"{last:.3f} at the final layer; the prediction never sharpened"
        )
