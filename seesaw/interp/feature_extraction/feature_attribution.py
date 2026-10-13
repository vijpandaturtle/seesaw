"""Feature attribution: which SAE features drive the logit diff?

Decompose the residual stream at the SAE's hook point into features, then
score each feature by activation × gradient (Marks et al. 2024):

    attr(f) = Σ_pos  act_f(pos) · ( W_dec[f] · ∂metric/∂resid(pos) )

— the first-order estimate of how much the metric would drop if the feature
were switched off. One forward and one backward pass covers every feature.
Like attribution patching, it's an estimate: `steering.feature_clamping`
checks the top features for real.

When the SAE is hosted on Neuronpedia, the top features come back with their
explanations and dashboard links attached.

Answers: "Which interpretable features is the model using for this output?"
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from transformer_lens import HookedTransformer

matplotlib.use("Agg")

from .._helpers import tokens_to_ids
from ..metrics import last_positions, logit_diff
from ..result import InterpResult
from . import neuronpedia
from .sae import hook_name, neuronpedia_source


def compute(
    model: HookedTransformer,
    sae,
    prompts: list[str],
    positive_tokens: list[str],
    negative_tokens: list[str],
    output_dir: Path,
    top_k: int = 10,
    explain: bool = True,
) -> InterpResult:
    """Score every SAE feature's contribution to the logit diff.

    Args:
        model: Loaded HookedTransformer.
        sae: SAE (see feature_extraction.sae.load_sae) on one of the model's hook points.
        prompts: Prompts the behaviour shows up on.
        positive_tokens: Correct answer token per prompt.
        negative_tokens: Distractor token per prompt.
        output_dir: Directory to save the plot.
        top_k: How many features to report (by attribution, positive and negative).
        explain: Look the top features up on Neuronpedia. Failures leave the
            explanation empty rather than failing the run.

    Returns:
        InterpResult with the top positive and negative features.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens = model.to_tokens(prompts)
    pos = last_positions(model, prompts)
    pos_ids = tokens_to_ids(model, positive_tokens)
    neg_ids = tokens_to_ids(model, negative_tokens)
    name = hook_name(sae)

    saved: dict[str, torch.Tensor] = {}

    def root(act, hook):
        return act.detach().requires_grad_(True)

    def keep(act, hook):
        saved["resid"] = act
        return act

    with torch.enable_grad():
        with model.hooks(fwd_hooks=[("hook_embed", root), (name, keep)]):
            logits = model(tokens)
        metric = logit_diff(logits, pos_ids, neg_ids, pos).mean()
        (grad,) = torch.autograd.grad(metric, saved["resid"])
    resid = saved["resid"].detach()

    with torch.no_grad():
        acts = sae.encode(resid.to(sae.W_dec.dtype))                 # [b, s, n_features]
        grad_dec = grad.to(sae.W_dec.dtype) @ sae.W_dec.T            # [b, s, n_features]
        # Positions after each prompt's end are padding; they must not count.
        seq = torch.arange(tokens.shape[1], device=tokens.device)
        mask = (seq[None, :] <= pos.to(tokens.device)[:, None]).to(acts.dtype)
        # grad already carries the batch mean's 1/batch, so summing over
        # prompts gives the mean per-prompt attribution.
        attr = (acts * grad_dec * mask[..., None]).sum(dim=(0, 1)).cpu()     # [n_features]
        mean_act = ((acts * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)).mean(0).cpu()

    source = neuronpedia_source(sae)

    def describe(idx: int) -> dict:
        entry = {"index": idx, "attribution": round(attr[idx].item(), 4),
                 "mean_activation": round(mean_act[idx].item(), 4)}
        if source:
            entry["url"] = neuronpedia.feature_url(source, idx)
            if explain:
                try:
                    entry["explanation"] = neuronpedia.get_feature(source, idx)["explanation"]
                except neuronpedia.NeuronpediaError:
                    entry["explanation"] = None
        return entry

    top_pos = [describe(i) for i in attr.topk(top_k).indices.tolist() if attr[i] > 0]
    top_neg = [describe(i) for i in (-attr).topk(top_k).indices.tolist() if attr[i] < 0]

    # ── Plot ──────────────────────────────────────────────────────────────────
    shown = top_pos[: top_k] + top_neg[: max(3, top_k // 3)]
    fig, ax = plt.subplots(figsize=(9, 0.35 * len(shown) + 1.5))
    labels = [
        f"#{f['index']}  {(f.get('explanation') or '')[:45]}" for f in shown
    ]
    values = [f["attribution"] for f in shown]
    ax.barh(range(len(shown)), values, color=["#d73027" if v > 0 else "#4575b4" for v in values])
    ax.set_yticks(range(len(shown)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Estimated contribution to logit diff")
    ax.set_title(f"Feature Attribution — {source or name}")
    plt.tight_layout()
    plot_path = output_dir / "feature_attribution.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return InterpResult(
        plot_paths=[plot_path],
        data={
            "hook_name":          name,
            "neuronpedia_source": source,
            "baseline_ld":        round(metric.item(), 4),
            "top_features":       top_pos,
            "top_negative_features": top_neg,
            "total_attribution":  round(attr.sum().item(), 4),
        },
    )
