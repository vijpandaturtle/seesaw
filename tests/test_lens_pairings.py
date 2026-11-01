"""expand_pairings: a plan's separate prompt and image lists become one image per prompt.

The case that motivated it: Scout wrote "3 prompts, 8 images" meaning every
pairing, and every lever tool refused it ("got 8 for 3 prompts").
"""

from __future__ import annotations

import pytest

tools = pytest.importorskip("lens.mcp_server.src.tools")
expand = tools.expand_pairings


def test_every_prompt_meets_every_image_prompt_major():
    prompts, kw = expand(["p1", "p2"], {"images": ["a", "b", "c"]})
    assert prompts == ["p1", "p1", "p1", "p2", "p2", "p2"]
    assert kw["images"] == ["a", "b", "c", "a", "b", "c"]


def test_one_prompt_many_images():
    prompts, kw = expand(["what colour?"], {"image": ["solid:red", "solid:blue"]})   # alias resolved
    assert prompts == ["what colour?"] * 2 and kw["images"] == ["solid:red", "solid:blue"]


def test_per_image_lists_tile_and_per_prompt_lists_repeat():
    prompts, kw = expand(
        ["p1", "p2"],
        {"images": ["solid:red", "solid:blue", "solid:green"],
         "corrupted_images": ["solid:blue", "solid:red", "solid:yellow"],
         "positive_tokens": ["red", "blue", "green"],        # one per image
         "negative_tokens": ["x", "y"]},                       # one per prompt
    )
    assert kw["corrupted_images"] == ["solid:blue", "solid:red", "solid:yellow"] * 2
    assert kw["positive_tokens"] == ["red", "blue", "green"] * 2
    assert kw["negative_tokens"] == ["x", "x", "x", "y", "y", "y"]
    # Still aligned: every pair keeps its own image's answer.
    assert list(zip(kw["images"], kw["positive_tokens"]))[4] == ("solid:blue", "blue")


@pytest.mark.parametrize("images", [["a", "b"], ["a"], "a", None])
def test_paired_single_or_absent_images_are_left_alone(images):
    kwargs = {} if images is None else {"images": images}
    prompts, kw = expand(["p1", "p2"], kwargs)
    assert prompts == ["p1", "p2"] and kw == kwargs


def test_expanded_spec_passes_normalisation(monkeypatch):
    monkeypatch.setenv("SEESAW_INTERP_BACKEND", "lever")
    prompts, kw = expand(["p1", "p2", "p3"], {"images": [f"solid:c{i}" for i in range(8)],
                                              "positive_tokens": ["red"], "negative_tokens": ["blue"]})
    out, missing, _ = tools.normalize_tool_kwargs("vision_ablation", kw, len(prompts))
    assert not missing and len(out["images"]) == len(prompts) == 24
    assert out["positive_tokens"] == ["red"] * 24
