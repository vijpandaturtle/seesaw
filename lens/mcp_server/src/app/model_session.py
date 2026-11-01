from torch import nn
from transformer_lens import HookedTransformer

from ..config import (
    TL_CENTER_WRITING_WEIGHTS,
    TL_CENTER_UNEMBED,
    TL_FOLD_LN,
    TL_REFACTOR_FACTORED_ATTN,
)

from shared.lens_tools import active_backend

# In-process model cache — avoids reloading weights on every experiment.
# Keyed by (backend, name): the same checkpoint loads differently in each.
_MODEL_CACHE: dict[tuple[str, str], object] = {}


class ModelRetired(RuntimeError):
    """Raised inside a timed-out experiment's thread when it next runs the model."""


def retire_model(model_name: str, model) -> None:
    """Take a model away from an experiment that timed out, and stop that experiment.

    The sandbox can't kill the timed-out thread, and it keeps running on the
    shared model, with its hooks attached. The next experiment then runs
    through those hooks: a stuck image_token_ablation's mask was applied to
    another prompt's activations ("mask [1, 749] … tensor [1, 745, …]").

    So the model is dropped from the cache (the next get_model loads a clean
    copy), and every module of the old one gets a pre-hook that raises. The
    stuck thread fails at its next forward call, its `with` blocks remove the
    hooks it added, and the old copy is freed once the thread ends. Until
    then both copies are in memory, which matters for models near the limit.
    """
    for key, cached in list(_MODEL_CACHE.items()):
        if key[1] == model_name and cached is model:
            del _MODEL_CACHE[key]
    # A TransformerLens model is the module; a lever handle wraps one in .model.
    root = model if isinstance(model, nn.Module) else getattr(model, "model", None)
    if not isinstance(root, nn.Module):
        return

    def refuse(_module, _args):
        raise ModelRetired(f"{model_name} was retired after its experiment timed out")

    for module in root.modules():
        module.register_forward_pre_hook(refuse)


def get_model(model_name: str):
    """Load the model for the active interp backend, caching it after first load.

    seesaw: a TransformerLens HookedTransformer. lever: a lever handle around
    the unmodified Hugging Face model (seesaw.vlm.load picks VLM or CLIP).

    Note: use_attn_result is intentionally NOT set — it conflicts with
    refactor_factored_attn_matrices. Tools that need per-head outputs use
    cache["z", layer] @ model.W_O[layer] instead of cache["result"].

    Args:
        model_name: HuggingFace / TransformerLens model name (e.g. 'gpt2', 'pythia-160m')

    Returns:
        The loaded, cached model in eval mode.
    """
    backend = active_backend()
    key = (backend, model_name)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if backend == "lever":
        from seesaw import vlm

        print(f"⏳ Loading {model_name} through lever...")
        _MODEL_CACHE[key] = vlm.load(model_name)
        print(f"✅ {model_name} — {type(_MODEL_CACHE[key]).__name__}")
        return _MODEL_CACHE[key]
    print(f"⏳ Loading {model_name}...")
    model = HookedTransformer.from_pretrained(
        model_name,
        center_writing_weights=TL_CENTER_WRITING_WEIGHTS,
        center_unembed=TL_CENTER_UNEMBED,
        fold_ln=TL_FOLD_LN,
        refactor_factored_attn_matrices=TL_REFACTOR_FACTORED_ATTN,
    )
    model.eval()
    _MODEL_CACHE[key] = model
    print(
        f"✅ {model_name} — "
        f"{model.cfg.n_layers}L {model.cfg.n_heads}H d_model={model.cfg.d_model}"
    )
    return model
