"""IOI: the baseline logit difference should be clearly positive.

On the standard IOI template GPT-2 small prefers the indirect object over the
subject by roughly 3 logits (Wang et al., 2022). The exact value moves with the
prompt set, so this checks the sign and a generous range rather than a number:
a baseline near zero means the prompts don't actually elicit the behaviour, and
everything measured against that baseline is noise.

That failure is not hypothetical — a gender-bias run in this repo produced a
baseline of -0.049, which made its ablation deltas meaningless.
"""

from conftest import require_tool

# Wide on purpose. The point is "the behaviour is present", not a fixed value.
MIN_BASELINE = 0.5
MAX_BASELINE = 10.0


def _baselines(by_tool) -> list[tuple[str, float]]:
    """(tool, baseline logit diff) for every tool that reports one."""
    found = []
    for result in by_tool.get("ablation", []):
        ld = result.get("data", {}).get("baseline_ld")
        if ld is not None:
            found.append(("ablation", float(ld)))
    for result in by_tool.get("activation_patching", []):
        ld = result.get("data", {}).get("clean_ld")
        if ld is not None:
            found.append(("activation_patching", float(ld)))
    return found


def test_a_baseline_was_measured(by_tool):
    """Ablation or patching has to have run for there to be a baseline."""
    if not _baselines(by_tool):
        require_tool(by_tool, "ablation")      # fails with the useful message
        raise AssertionError("ablation ran but reported no baseline_ld")


def test_baseline_is_positive_and_plausible(by_tool):
    for tool, ld in _baselines(by_tool):
        assert ld > MIN_BASELINE, (
            f"{tool} baseline logit diff is {ld:+.4f}; the model barely prefers "
            f"the indirect object, so these prompts don't elicit IOI and any "
            f"intervention measured against this baseline is noise"
        )
        assert ld < MAX_BASELINE, (
            f"{tool} baseline logit diff is {ld:+.4f}, implausibly large for "
            f"GPT-2 small on IOI — check the token pair"
        )


def test_corrupted_run_flips_the_sign(by_tool):
    """Patching needs a corrupted prompt the model gets wrong. If the corrupt
    logit diff isn't below the clean one, the corruption did nothing and the
    recovery numbers are meaningless."""
    for result in by_tool.get("activation_patching", []):
        data = result.get("data", {})
        clean, corrupt = data.get("clean_ld"), data.get("corrupt_ld")
        if clean is None or corrupt is None:
            continue
        assert corrupt < clean, (
            f"corrupted logit diff {corrupt:+.4f} is not below clean "
            f"{clean:+.4f} — the corrupted prompts aren't corrupting anything"
        )
