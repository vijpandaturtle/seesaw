"""IOI: direct logit attribution should surface the name mover heads.

Wang et al. (2022) identify L9H9 and L9H6 as the primary name movers in
GPT-2 small — the heads that copy the indirect object's name into the final
logits — with L10H7 and L11H10 acting as negative name movers that push the
other way.

A DLA sweep that doesn't put a name mover near the top either ran on the wrong
prompts, used the wrong token pair, or isn't measuring what it claims to.
"""

from conftest import head_names, head_scores, layer_of, require_tool

NAME_MOVERS = {"L9H9", "L9H6"}
NEGATIVE_NAME_MOVERS = {"L10H7", "L11H10"}


def test_dla_ran(by_tool):
    require_tool(by_tool, "direct_logit_attribution")


def test_top_heads_include_a_name_mover(by_tool):
    """At least one of L9H9/L9H6 in the top 5 by |attribution|."""
    for result in require_tool(by_tool, "direct_logit_attribution"):
        top = head_names(result)
        if NAME_MOVERS & set(top):
            return
    raise AssertionError(
        f"no name mover {sorted(NAME_MOVERS)} in the top DLA heads; got {top}"
    )


def test_strongest_positive_head_is_late_layer(by_tool):
    """Name movers live in layers 9-11; a top positive head in an early layer
    means the sweep is picking up something other than name movement."""
    for result in require_tool(by_tool, "direct_logit_attribution"):
        scores = head_scores(result)
        positive = {h: s for h, s in scores.items() if s > 0}
        if not positive:
            continue
        strongest = max(positive, key=positive.get)
        assert layer_of(strongest) >= 8, (
            f"strongest positive DLA head {strongest} is in layer "
            f"{layer_of(strongest)}; name movers are layers 9-11"
        )


def test_negative_name_movers_have_negative_attribution(by_tool):
    """L10H7 and L11H10 suppress the IO token, so where they appear their
    attribution must be negative — a positive value means the sign convention
    is inverted somewhere."""
    for result in require_tool(by_tool, "direct_logit_attribution"):
        for head, score in head_scores(result).items():
            if head in NEGATIVE_NAME_MOVERS:
                assert score < 0, (
                    f"{head} is a negative name mover but scored {score:+.4f}; "
                    f"check the positive/negative token order"
                )
