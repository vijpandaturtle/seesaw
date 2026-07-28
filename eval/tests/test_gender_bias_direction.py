"""Gender bias: stereotyped prompts should show a measurable, signed effect.

Vig et al. (2020) find gender bias in GPT-2 is sparse and localised, with
particular heads mediating the he/she preference on profession prompts.

The precondition for saying anything at all is a baseline that actually leans
one way. A near-zero logit difference means the prompts didn't elicit the bias,
and every intervention measured against it is noise — this repo produced
exactly that failure, a baseline of -0.049, and the ablation deltas from that
run meant nothing.
"""

from conftest import head_scores, require_tool

# Below this the prompts aren't eliciting a preference worth measuring.
MIN_ABS_BASELINE = 0.3


def _baselines(by_tool) -> list[tuple[str, float]]:
    found = []
    for r in by_tool.get("ablation", []):
        ld = r.get("data", {}).get("baseline_ld")
        if ld is not None:
            found.append(("ablation", float(ld)))
    for r in by_tool.get("activation_patching", []):
        ld = r.get("data", {}).get("clean_ld")
        if ld is not None:
            found.append(("activation_patching", float(ld)))
    return found


def test_a_causal_experiment_ran(by_tool):
    """Attention patterns alone are correlational; the bias claim needs an
    intervention."""
    if not (by_tool.get("ablation") or by_tool.get("activation_patching")):
        require_tool(by_tool, "ablation")


def test_baseline_shows_a_real_preference(by_tool):
    for tool, ld in _baselines(by_tool):
        assert abs(ld) >= MIN_ABS_BASELINE, (
            f"{tool} baseline logit diff is {ld:+.4f}, effectively neutral. "
            f"These prompts don't elicit a gendered preference, so any head "
            f"effect measured against this baseline is noise"
        )


def test_intervention_moves_the_prediction(by_tool):
    """If ablating every head changes nothing, no head mediates the behaviour
    and the sweep found nothing to report."""
    for result in by_tool.get("ablation", []):
        effects = head_scores(result)
        if not effects:
            continue
        largest = max(abs(v) for v in effects.values())
        assert largest > 0.01, (
            f"the largest ablation effect is {largest:.4f}; no head measurably "
            f"changes the prediction, so nothing was localised"
        )
