"""The lever backend: seesaw.vlm on tiny random-weight VLMs and CLIP, and Lens dispatch with SEESAW_INTERP_BACKEND=lever.

Tiny models have no processor, so `prepare` is swapped for fixed inputs whose
image tokens sit in known positions, and answer "tokens" are ids. That's
enough to check the mechanics exactly: an edit that should change nothing
changes nothing, an edit that should remove everything removes everything, and
CLIP's direct effects add up to the real logits. Whether the results mean
anything on a real model is for a real model.

Needs the [lever] extra; skipped without it.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lever")
transformers = pytest.importorskip("transformers")

from lever import CLIP, label_probs, wrap_model  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import CLIPConfig, CLIPModel, LlavaConfig, LlavaForConditionalGeneration  # noqa: E402

from seesaw.vlm import clip as vclip  # noqa: E402
from seesaw.vlm import inputs as vinputs  # noqa: E402
from seesaw.vlm import interventions, logit_lens  # noqa: E402

TEXT = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2)
TINY_VIT = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, image_size=32,
                patch_size=8)
PROMPTS = ["what is this?", "is it red?"]
POS, NEG = ["5"], ["6"]


@pytest.fixture(scope="module")
def vlm():
    torch.manual_seed(0)
    cfg = LlavaConfig(vision_config=TINY_VIT | dict(model_type="clip_vision_model"),
                      text_config=TEXT | dict(model_type="llama", vocab_size=32064), image_token_id=32000)
    return wrap_model(LlavaForConditionalGeneration(cfg).eval())


@pytest.fixture(autouse=True)
def fixed_inputs(monkeypatch, vlm):
    """Inputs for the tiny LLaVA: 16 image tokens between text, pixels seeded by the image's colour,
    so different images give different (but repeatable) inputs."""
    vc = vlm.model.config.vision_config
    n_img = (vc.image_size // vc.patch_size) ** 2

    def prepare(_vlm, image, prompt):
        seed = int.from_bytes(bytes(image.getpixel((0, 0)))) + len(prompt)
        g = torch.Generator().manual_seed(seed)
        ids = torch.tensor([[1, 2] + [vlm.model.config.image_token_id] * n_img + [3, 4, len(prompt) % 50 + 7]])
        return dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                    pixel_values=torch.randn(1, 3, vc.image_size, vc.image_size, generator=g))

    monkeypatch.setattr(vinputs, "prepare", prepare)
    # A random model's answer barely depends on the image; the gap guard is for real models.
    monkeypatch.setattr(interventions, "MIN_GAP", 1e-6)
    for mod in (logit_lens, interventions):
        monkeypatch.setattr(mod, "prepare", prepare)


@pytest.fixture(scope="module")
def tiny_clip():
    torch.manual_seed(0)
    cfg = CLIPConfig(text_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2),
                     vision_config=TINY_VIT | dict(num_attention_heads=4), projection_dim=16)
    model = CLIPModel(cfg).eval()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "layer_norm" in n or "layrnorm" in n or n.endswith("bias"):
                p.add_(0.1 * torch.randn_like(p))
    clip = CLIP(model, None)
    labels = torch.nn.functional.normalize(torch.randn(3, 16, generator=torch.Generator().manual_seed(1)), dim=-1)
    clip.pixels = lambda image: torch.randn(1, 3, 32, 32, generator=torch.Generator().manual_seed(
        int.from_bytes(bytes(image.getpixel((0, 0))))))
    clip.encode_text = lambda texts: labels[: len(texts)]
    return clip


# ── image references ──────────────────────────────────────────────────────────


def test_image_references(tmp_path, monkeypatch):
    assert vinputs.load_image("solid:red").getpixel((5, 5)) == (255, 0, 0)
    assert vinputs.load_image("text:STOP").getpixel((0, 0)) == (255, 255, 255)
    Image.new("RGB", (40, 30), "blue").save(tmp_path / "b.png")
    monkeypatch.setenv("SEESAW_IMAGE_DIR", str(tmp_path))
    plain, stamped = vinputs.load_image("b.png"), vinputs.load_image("b.png+text:cat")
    assert plain.size == stamped.size == (40, 30) and plain.tobytes() != stamped.tobytes()
    with pytest.raises(FileNotFoundError):
        vinputs.load_image("missing.png")


def test_broadcast():
    assert vinputs.broadcast("a", 3, "image") == ["a", "a", "a"]
    with pytest.raises(ValueError):
        vinputs.broadcast(["a", "b"], 3, "image")


# ── VLM tools ─────────────────────────────────────────────────────────────────


def test_logit_lens_last_layer_is_the_model_output(vlm, tmp_path):
    r = logit_lens.compute(vlm, PROMPTS, ["solid:red"], POS, NEG, tmp_path)
    assert len(r.data["layer_logit_diffs"]) == vlm.n_layers
    assert r.data["layer_logit_diffs"][-1] == pytest.approx(r.data["final_logit_diff"], abs=1e-3)
    assert r.data["image_token_top_tokens"]
    json.dumps(r.data)


def test_vision_ablation_changes_the_answer(vlm, tmp_path):
    r = interventions.vision_ablation(vlm, PROMPTS, ["solid:red", "solid:blue"], POS, NEG, tmp_path,
                                      ablation_type="zero")
    assert r.data["ablated_logit_diff"] != pytest.approx(r.data["clean_logit_diff"])
    assert all(p.exists() for p in r.plot_paths)


def test_image_token_ablation_after_the_last_layer_changes_nothing(vlm, tmp_path):
    """Wiping image positions after the last layer can't reach the answer position's output."""
    r = interventions.image_token_ablation(vlm, PROMPTS, ["solid:red"], POS, NEG, tmp_path)
    assert r.data["layer_logit_diffs"][-1] == pytest.approx(r.data["clean_logit_diff"], abs=1e-4)


def test_patching_every_position_at_the_last_layer_recovers_everything(vlm, tmp_path):
    r = interventions.activation_patching(vlm, PROMPTS, ["solid:red"], ["solid:blue"], POS, NEG, tmp_path)
    rec = r.data["recovery"]
    assert rec["answer"][-1] == pytest.approx(1.0, abs=1e-3)   # the last layer at the answer is the output
    assert rec["image"][-1] == pytest.approx(0.0, abs=1e-3)    # ...and nothing else reaches it there


def test_patching_identical_images_has_nothing_to_recover(vlm, tmp_path):
    with pytest.raises(ValueError, match="nothing to recover"):
        interventions.activation_patching(vlm, PROMPTS, ["solid:red"], ["solid:red"], POS, NEG, tmp_path)


# ── CLIP tools ────────────────────────────────────────────────────────────────


def test_clip_direct_effects_add_up_to_the_label_logits(tiny_clip, tmp_path):
    from lever import decompose_image_embedding

    r = vclip.direct_effects(tiny_clip, ["a", "b", "c"], ["solid:red", "solid:green"], tmp_path)
    assert len(r.data["head_grid"]) == tiny_clip.n_layers
    # The prediction read off the summed contributions is the model's own.
    img = vinputs.load_image("solid:red")
    probs = label_probs(tiny_clip, tiny_clip.encode_image(tiny_clip.pixels(img)), tiny_clip.encode_text(["a", "b", "c"]))[0]
    assert r.data["predictions"][0]["label"] == "abc"[int(probs.argmax())]
    assert r.data["predictions"][0]["prob"] == pytest.approx(probs.max().item(), abs=1e-3)
    comps, _ = decompose_image_embedding(tiny_clip, tiny_clip.pixels(img))
    emb = tiny_clip.model.visual_projection(tiny_clip.vision(pixel_values=tiny_clip.pixels(img)).pooler_output)
    assert torch.allclose(comps.sum(1), emb, atol=1e-4)


def test_clip_head_ablation(tiny_clip, tmp_path):
    r = vclip.head_ablation(tiny_clip, ["a", "b", "c"], ["solid:red"], ["L0H1", "L1H3"], tmp_path)
    assert set(r.data["prob_drop"]) == {"L0H1", "L1H3", "all together"}
    with pytest.raises(ValueError):
        vclip.head_ablation(tiny_clip, ["a"], ["solid:red"], ["L9H0"], tmp_path)


# ── Lens dispatch with the lever backend ──────────────────────────────────────


lens_tools = pytest.importorskip("lens.mcp_server.src.tools")


@pytest.fixture
def lever_backend(monkeypatch):
    monkeypatch.setenv("SEESAW_INTERP_BACKEND", "lever")


PLAN_KWARGS = {"positive_tokens": POS, "negative_tokens": NEG, "image": "solid:red",
               "counterfactual_images": ["solid:blue"], "heads": ["L0H1"]}
VLM_TOOLS = ["logit_lens", "vision_ablation", "image_token_ablation", "activation_patching"]


@pytest.mark.parametrize("tool", VLM_TOOLS + ["clip_direct_effects", "clip_head_ablation"])
def test_every_lever_tool_runs_through_dispatch(tool, lever_backend, vlm, tiny_clip, tmp_path):
    from lens.mcp_server.src.app.sandbox import run_in_sandbox

    kwargs, missing, _ = lens_tools.normalize_tool_kwargs(tool, PLAN_KWARGS, len(PROMPTS))
    assert not missing, missing
    assert kwargs["images"] == ["solid:red", "solid:red"]          # aliased and broadcast
    model, prompts = (vlm, PROMPTS) if tool in VLM_TOOLS else (tiny_clip, ["a", "b", "c"])
    result = run_in_sandbox(
        lens_tools.registry()[tool],
        tool_kwargs={"model": model, "prompts": prompts, "output_dir": tmp_path,
                     **{k: v[:1] * len(prompts) if k.endswith("images") else v for k, v in kwargs.items()}},
        experiment_name=tool, tool=tool,
    )
    assert result.status == "success", result.error
    json.dumps(result.data)


def test_a_vlm_tool_refuses_clip(lever_backend, tiny_clip, tmp_path):
    from lens.mcp_server.src.tools.lever_tools import run_vision_ablation

    with pytest.raises(TypeError, match="vision-language"):
        run_vision_ablation(tiny_clip, ["a"], ["solid:red"], POS, NEG, tmp_path)
