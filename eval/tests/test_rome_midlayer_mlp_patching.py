"""Factual recall: the causal signal should sit in mid-layer MLPs.

ROME (Meng et al., 2022) localises factual associations to mid-layer MLP
modules at the subject's final token — patching there restores the fact, while
patching late layers does not. On GPT-2 small that region is roughly layers
3-8.

Activation patching reports recovery per (layer, position), so this checks the
peak is in the middle of the network rather than at either end.
"""

from conftest import require_tool

MIDLAYER_FRACTION = (0.15, 0.75)   # of total depth
MIN_PEAK_RECOVERY = 0.20           # below this, nothing was localised at all


def _peak_layer(result: dict) -> tuple[int, float] | None:
    """(layer, recovery) for the strongest patch effect."""
    effects = result.get("data", {}).get("patch_effects")
    if not effects:
        return None
    best_layer, best_value = 0, float("-inf")
    for layer, row in enumerate(effects):
        if not row:
            continue
        peak = max(row)
        if peak > best_value:
            best_layer, best_value = layer, peak
    return best_layer, best_value


def test_activation_patching_ran(by_tool):
    require_tool(by_tool, "activation_patching")


def test_something_was_localised(by_tool):
    """A flat recovery map means the patch never restored the fact."""
    peaks = [
        _peak_layer(r) for r in require_tool(by_tool, "activation_patching")
    ]
    peaks = [p for p in peaks if p]
    assert peaks, "activation patching produced no patch_effects grid"
    best = max(v for _, v in peaks)
    assert best >= MIN_PEAK_RECOVERY, (
        f"peak recovery is only {best:.3f}; patching restored almost nothing, "
        f"so no component was localised"
    )


def test_peak_is_in_middle_layers(by_tool):
    """ROME's finding: mid-layer MLPs, not the first or last layers."""
    for result in require_tool(by_tool, "activation_patching"):
        peak = _peak_layer(result)
        effects = result.get("data", {}).get("patch_effects") or []
        if peak is None or not effects:
            continue
        layer, value = peak
        if value < MIN_PEAK_RECOVERY:
            continue        # covered by the test above
        depth = len(effects)
        low, high = (int(depth * f) for f in MIDLAYER_FRACTION)
        assert low <= layer <= high, (
            f"peak recovery {value:.3f} is at layer {layer} of {depth}; "
            f"factual recall localises to mid layers ({low}-{high})"
        )
