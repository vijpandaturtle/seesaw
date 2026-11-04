"""Model diffing: what changed between two checkpoints, on which prompts, and in which weights?

For a base model A and a changed model B (fine-tuned, quantised, edited):

1. Behaviour: the KL divergence of B's next-token distribution from A's at
   every prompt's final position. Large KL on a prompt = B behaves
   differently there.
2. Direction: amplified logits, B + α(B − A), exaggerate what B does
   differently (logit-diff amplification); the tokens that rise most, and
   which tokens gained probability on average, say what the change is.
3. Location (same architecture only): swap one component's weights from B
   into A (the embeddings, one block, or the unembedding) and see how much
   of the KL that closes. A change that lives in one block closes most of
   it from that block alone; a diffuse one doesn't close much from any.

Answers: "What did this fine-tune (or quantisation) change, and where?"
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from transformer_lens import HookedTransformer

matplotlib.use("Agg")

from ..metrics import last_positions
from ..result import InterpResult


def same_architecture(a: HookedTransformer, b: HookedTransformer) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(sa[k].shape == sb[k].shape for k in sa)


def _final_logprobs(model: HookedTransformer, prompts: list[str]) -> torch.Tensor:
    tokens = model.to_tokens(prompts)
    pos = last_positions(model, prompts).to(tokens.device)
    with torch.no_grad():
        logits = model(tokens)
    return logits[torch.arange(len(prompts), device=tokens.device), pos].float().log_softmax(-1).cpu()


def _kl(p_log: torch.Tensor, q_log: torch.Tensor) -> torch.Tensor:
    """Per-row KL(P || Q) from log-probabilities."""
    return (p_log.exp() * (p_log - q_log)).sum(-1)


def components(model: HookedTransformer) -> dict[str, list[str]]:
    """Component name -> the state-dict keys that make it up."""
    keys = list(model.state_dict())
    out = {"embed": [k for k in keys if k.startswith(("embed.", "pos_embed."))]}
    for l in range(model.cfg.n_layers):
        out[f"block{l}"] = [k for k in keys if k.startswith(f"blocks.{l}.")]
    out["unembed"] = [k for k in keys if k.startswith(("unembed.", "ln_final."))]
    return {name: ks for name, ks in out.items() if ks}


@contextlib.contextmanager
def swapped(a: HookedTransformer, b: HookedTransformer, keys: list[str]):
    """A with B's weights for `keys`; A's own weights are put back on exit."""
    pa, pb = a.state_dict(), b.state_dict()
    originals = {k: pa[k].clone() for k in keys}
    with torch.no_grad():
        for k in keys:
            pa[k].copy_(pb[k].to(pa[k].device, pa[k].dtype))
    try:
        yield a
    finally:
        with torch.no_grad():
            for k, v in originals.items():
                pa[k].copy_(v)


def compute(
    model: HookedTransformer,
    other_model: HookedTransformer,
    prompts: list[str],
    output_dir: Path,
    alpha: float = 1.0,
    top_k: int = 10,
) -> InterpResult:
    """Compare `model` (A, the base) with `other_model` (B, the changed one) on `prompts`.

    Args:
        model: The base model A.
        other_model: The changed model B, with the same tokenizer.
        prompts: Prompts to compare them on.
        output_dir: Directory to save the plot.
        alpha: Amplification strength for B + alpha (B - A).
        top_k: Tokens and prompts to report.

    Returns:
        InterpResult with per-prompt KL(B || A), the most changed prompts, the
        tokens B favours, and (same architecture only) the share of the KL
        each component's weights close when swapped in alone.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    la, lb = _final_logprobs(model, prompts), _final_logprobs(other_model, prompts)
    kl = _kl(lb, la)
    mean_kl = kl.mean().item()

    order = kl.argsort(descending=True)[:top_k].tolist()
    amplified = (lb + alpha * (lb - la)).log_softmax(-1)

    def tok(i: int) -> str:
        return model.to_string([i])

    most_changed = [{
        "prompt": prompts[i],
        "kl": round(kl[i].item(), 4),
        "top_a": tok(int(la[i].argmax())),
        "top_b": tok(int(lb[i].argmax())),
        "top_amplified": [tok(int(t)) for t in amplified[i].topk(3).indices],
    } for i in order]
    gained = (lb.exp() - la.exp()).mean(0)
    favoured = [{"token": tok(int(i)), "gain": round(gained[i].item(), 5)}
                for i in gained.topk(top_k).indices]
    disfavoured = [{"token": tok(int(i)), "gain": round(gained[i].item(), 5)}
                   for i in (-gained).topk(top_k).indices]

    closed = None
    if same_architecture(model, other_model) and mean_kl > 1e-8:
        closed = {}
        for name, keys in components(model).items():
            with swapped(model, other_model, keys):
                after = _kl(lb, _final_logprobs(model, prompts)).mean().item()
            closed[name] = round(1 - after / mean_kl, 4)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2 if closed else 1, figsize=(12 if closed else 6, 4), squeeze=False)
    axes[0][0].bar(range(len(kl)), kl.tolist(), color="#d73027")
    axes[0][0].set_xlabel("prompt")
    axes[0][0].set_ylabel("KL(B || A), nats")
    axes[0][0].set_title(f"Behaviour change — mean KL {mean_kl:.3f}")
    if closed:
        ax = axes[0][1]
        ax.bar(range(len(closed)), list(closed.values()), color="#4575b4")
        ax.set_xticks(range(len(closed)))
        ax.set_xticklabels(list(closed), rotation=60, fontsize=7)
        ax.set_ylabel("share of KL closed")
        ax.set_title("Swapping one component from B into A")
    plt.tight_layout()
    plot_path = output_dir / "model_diff.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return InterpResult(
        plot_paths=[plot_path],
        data={
            "kl_per_prompt":   [round(v, 4) for v in kl.tolist()],
            "mean_kl":         round(mean_kl, 6),
            "most_changed":    most_changed,
            "favoured_tokens": favoured,
            "disfavoured_tokens": disfavoured,
            "same_architecture": closed is not None or same_architecture(model, other_model),
            "kl_closed_by_component": closed,
        },
    )
