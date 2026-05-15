from .logit_lens import run_logit_lens
from .attention_pattern import run_attention_pattern
from .direct_logit_attribution import run_direct_logit_attribution
from .ablation import run_ablation
from .activation_patching import run_activation_patching

TOOL_REGISTRY: dict = {
    "logit_lens":               run_logit_lens,
    "attention_pattern":        run_attention_pattern,
    "direct_logit_attribution": run_direct_logit_attribution,
    "ablation":                 run_ablation,
    "activation_patching":      run_activation_patching,
    # Tier 2 (reasoning models): cot_faithfulness, thinking_token_lens, thought_intervention
    # Tier 3 (SAE): sae_feature_search, feature_steering
    # Tier 4 (safety): linear_probe, representation_reading
}

__all__ = [
    "run_logit_lens",
    "run_attention_pattern",
    "run_direct_logit_attribution",
    "run_ablation",
    "run_activation_patching",
    "TOOL_REGISTRY",
]
