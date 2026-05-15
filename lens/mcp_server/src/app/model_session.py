from transformer_lens import HookedTransformer

from ..config import (
    TL_CENTER_WRITING_WEIGHTS,
    TL_CENTER_UNEMBED,
    TL_FOLD_LN,
    TL_REFACTOR_FACTORED_ATTN,
)

# In-process model cache — avoids reloading weights on every experiment
_MODEL_CACHE: dict[str, HookedTransformer] = {}


def get_model(model_name: str) -> HookedTransformer:
    """Load a TransformerLens model, caching it after first load.

    Note: use_attn_result is intentionally NOT set — it conflicts with
    refactor_factored_attn_matrices. Tools that need per-head outputs use
    cache["z", layer] @ model.W_O[layer] instead of cache["result"].

    Args:
        model_name: HuggingFace / TransformerLens model name (e.g. 'gpt2', 'pythia-160m')

    Returns:
        Loaded and cached HookedTransformer model in eval mode.
    """
    if model_name not in _MODEL_CACHE:
        print(f"⏳ Loading {model_name}...")
        model = HookedTransformer.from_pretrained(
            model_name,
            center_writing_weights=TL_CENTER_WRITING_WEIGHTS,
            center_unembed=TL_CENTER_UNEMBED,
            fold_ln=TL_FOLD_LN,
            refactor_factored_attn_matrices=TL_REFACTOR_FACTORED_ATTN,
        )
        model.eval()
        _MODEL_CACHE[model_name] = model
        print(
            f"✅ {model_name} — "
            f"{model.cfg.n_layers}L {model.cfg.n_heads}H d_model={model.cfg.d_model}"
        )
    return _MODEL_CACHE[model_name]
