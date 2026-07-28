"""Gender bias: the study needs controls, not just stereotyped prompts.

A bias claim built only on stereotype-consistent prompts can't separate "this
head mediates gender bias" from "this head mediates pronoun prediction". The
control is counter-stereotypical or neutral prompts — Vig et al. (2020) rely on
exactly that contrast, and Quill flags its absence as a critical gap.

These checks are about study design rather than a published number: did the run
gather enough prompts, and did it intervene rather than only observe?
"""

from conftest import require_tool

MIN_PROMPTS = 3


def _all_prompts(bundle: dict) -> list[str]:
    seen = []
    for r in bundle.get("results", []):
        for p in r.get("prompts") or []:
            if p not in seen:
                seen.append(p)
    return seen


def test_enough_prompts_to_generalise(bundle):
    """One or two prompts can't distinguish a bias effect from a quirk of
    the specific sentence."""
    prompts = _all_prompts(bundle)
    assert len(prompts) >= MIN_PROMPTS, (
        f"the run used {len(prompts)} distinct prompt(s): {prompts}. "
        f"Too few to separate a general bias effect from one sentence's quirks"
    )


def test_a_causal_tool_was_used(by_tool):
    """Attention patterns and logit lens are observational. Naming heads as
    responsible for bias requires an intervention."""
    causal = {"ablation", "activation_patching"} & set(by_tool)
    assert causal, (
        f"only observational tools ran ({sorted(by_tool)}). Attributing bias "
        f"to specific heads needs ablation or activation patching"
    )


def test_patching_used_a_contrast_pair(by_tool):
    """Activation patching's corrupted prompts are the built-in control: the
    same sentence with the key entity swapped."""
    for result in by_tool.get("activation_patching", []):
        data = result.get("data", {})
        corrupted = data.get("corrupted_prompts") or []
        clean = result.get("prompts") or []
        assert corrupted, (
            "activation patching ran without corrupted_prompts recorded; "
            "without the contrast there is no control condition"
        )
        assert corrupted != clean, (
            "corrupted prompts are identical to the clean prompts, so the "
            "contrast is empty"
        )
