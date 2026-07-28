"""Successor heads: the same head should fire across ordinal types.

Gould et al. (2023) find successor heads that increment ordered sequences —
numbers, days, months — using a shared representation, so the same head is
implicated whichever sequence type you probe. In GPT-2 small the canonical one
is L9H1.

The cross-type claim needs more than one experiment to check, so this compares
the heads implicated across every sweep in the run and requires them to
overlap. A run where each prompt set implicates a disjoint set of heads is
evidence against a shared mechanism — or of an unstable measurement.
"""

from conftest import head_names, require_tool


def _sweeps(by_tool) -> list[dict]:
    return by_tool.get("direct_logit_attribution", []) + by_tool.get("ablation", [])


def test_a_head_sweep_ran(by_tool):
    if not _sweeps(by_tool):
        require_tool(by_tool, "direct_logit_attribution")


def test_sweeps_report_heads(by_tool):
    for result in _sweeps(by_tool):
        assert head_names(result), (
            f"{result.get('tool')} reported no top_heads, so nothing can be "
            f"compared across sequence types"
        )


def test_heads_overlap_across_experiments(by_tool):
    """Two or more sweeps must share at least one head.

    With a single sweep there's nothing to cross-check, and the test passes —
    the tool_calls grader is what enforces that enough experiments ran.
    """
    sets = [set(head_names(r)) for r in _sweeps(by_tool)]
    sets = [s for s in sets if s]
    if len(sets) < 2:
        return

    shared = set.intersection(*sets)
    assert shared, (
        "no head appears in the top set of every sweep: "
        + " vs ".join(sorted(s) and str(sorted(s)) for s in sets)
        + ". Successor heads use a shared representation across ordinal types, "
        "so the implicated heads should overlap"
    )
