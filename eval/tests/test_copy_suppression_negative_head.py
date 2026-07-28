"""Copy suppression: L10H7 should push against the token it attends to.

McDougall et al. (2023) characterise GPT-2 small's L10H7 as a copy suppression
head — it attends to an earlier occurrence of the current prediction and writes
*against* it, which is why it shows up with negative attribution in IOI.

The observable consequence in this suite's tools: a negative contribution from
L10H7, and ablating it should raise the logit difference rather than lower it.
"""

from conftest import head_scores, require_tool

COPY_SUPPRESSION_HEAD = "L10H7"


def test_a_head_sweep_ran(by_tool):
    if not (by_tool.get("direct_logit_attribution") or by_tool.get("ablation")):
        require_tool(by_tool, "ablation")


def test_l10h7_attribution_is_negative_where_present(by_tool):
    """If DLA surfaced L10H7 at all, its contribution must be negative."""
    for result in by_tool.get("direct_logit_attribution", []):
        score = head_scores(result).get(COPY_SUPPRESSION_HEAD)
        if score is not None:
            assert score < 0, (
                f"{COPY_SUPPRESSION_HEAD} scored {score:+.4f} in DLA; a copy "
                f"suppression head writes against the prediction, so this "
                f"should be negative"
            )


def test_ablating_l10h7_helps(by_tool):
    """Ablation records baseline - ablated. Removing a suppressor lets the
    prediction through, so the recorded effect is negative for L10H7."""
    for result in by_tool.get("ablation", []):
        effect = head_scores(result).get(COPY_SUPPRESSION_HEAD)
        if effect is not None:
            assert effect < 0, (
                f"ablating {COPY_SUPPRESSION_HEAD} changed the logit diff by "
                f"{effect:+.4f}; suppressing a suppressor should *raise* the "
                f"logit difference, giving a negative recorded effect"
            )


def test_some_head_has_a_negative_effect(by_tool):
    """A sweep where every head helps has no suppression in it at all, which
    contradicts the phenomenon under study."""
    for result in by_tool.get("direct_logit_attribution", []):
        scores = head_scores(result)
        if scores and not any(s < 0 for s in scores.values()):
            raise AssertionError(
                f"no head has negative attribution: {scores}. Copy suppression "
                f"predicts at least one head writing against the prediction"
            )
