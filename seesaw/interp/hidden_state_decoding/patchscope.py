"""Patchscope: read what a hidden state encodes by letting the model itself decode it.

The logit lens decodes a layer's residual stream with the final unembedding,
which reads early layers badly on bigger models: their representations aren't
in the output basis yet. A patchscope (Ghandeharioun et al. 2024) instead
patches the hidden state into a prompt built to make the model repeat what it
is given ("cat -> cat; 1135 -> 1135; hello -> hello; ?"), at the same layer,
and reads what the model says next. No training, so it works on any model;
a tuned lens needs a translator trained per model, and none are published for
the models Lens targets.

Both readings are reported per layer, so the gap between them shows where
the logit lens under-reads.

Answers: "From which layer does the model's hidden state already encode the
answer, read by the model itself rather than by the final unembedding?"
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from transformer_lens import HookedTransformer

matplotlib.use("Agg")

from .._helpers import tokens_to_ids
from ..metrics import last_positions
from ..result import InterpResult

# The identity prompt: few-shot "x -> x", ending where the patched state goes.
IDENTITY = "cat -> cat\n1135 -> 1135\nhello -> hello\n?"


def compute(
    model: HookedTransformer,
    prompts: list[str],
    output_dir: Path,
    positive_tokens: list[str] | None = None,
    target_prompt: str = IDENTITY,
    top_k: int = 5,
) -> InterpResult:
    """Decode each prompt's last-token hidden state at every layer, by patchscope and by logit lens.

    Args:
        model: Loaded HookedTransformer.
        prompts: Prompts whose final hidden states to read.
        output_dir: Directory to save the plot.
        positive_tokens: The answer per prompt, to track its probability by layer.
        target_prompt: The prompt the hidden state is patched into, at its last token.
        top_k: Tokens to list per layer (for the first prompt).

    Returns:
        InterpResult with, per layer, the mean answer probability under each
        reading, the first prompt's top tokens under each, and the first layer
        where the answer is the top token for most prompts under each.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n, n_layers = len(prompts), model.cfg.n_layers
    tokens = model.to_tokens(prompts)
    rows = torch.arange(n, device=tokens.device)
    pos = last_positions(model, prompts).to(tokens.device)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x.endswith("hook_resid_post"))
    states = [cache[f"blocks.{l}.hook_resid_post"][rows, pos] for l in range(n_layers)]   # [n, d]
    del cache
    target = model.to_tokens(target_prompt).repeat(n, 1)
    slot = target.shape[1] - 1
    answers = torch.tensor(tokens_to_ids(model, positive_tokens)) if positive_tokens else None

    def summarise(probs: torch.Tensor) -> dict:
        probs = probs.float().cpu()
        out = {"top_tokens": [model.to_string(int(t)) for t in probs[0].topk(top_k).indices]}
        if answers is not None:
            p = probs[torch.arange(n), answers]
            out["answer_prob"] = round(p.mean().item(), 4)
            out["answer_top1"] = round((probs.argmax(-1) == answers).float().mean().item(), 4)
        return out

    per_layer = []
    for layer, h in enumerate(states):
        def put(value, hook, h=h):
            value = value.clone()
            value[:, slot] = h
            return value

        with torch.no_grad(), model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_post", put)]):
            scope = model(target)[:, slot].softmax(-1)
            lens = model.unembed(model.ln_final(h[:, None]))[:, 0].softmax(-1)
        per_layer.append({"layer": layer, "patchscope": summarise(scope), "logit_lens": summarise(lens)})

    def first(kind: str) -> int | None:
        if answers is None:
            return None
        return next((r["layer"] for r in per_layer if r[kind]["answer_top1"] > 0.5), None)

    plot_paths = []
    if answers is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        for kind, color in (("patchscope", "#d73027"), ("logit_lens", "#4575b4")):
            ax.plot(range(n_layers), [r[kind]["answer_prob"] for r in per_layer], marker="o",
                    color=color, label=kind.replace("_", " "))
        ax.set_xlabel("Layer (resid_post)")
        ax.set_ylabel("P(answer), mean over prompts")
        ax.set_title("Patchscope vs logit lens")
        ax.legend()
        plt.tight_layout()
        path = output_dir / "patchscope.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plot_paths.append(path)

    return InterpResult(plot_paths=plot_paths, data={
        "per_layer": per_layer,
        "first_layer_patchscope": first("patchscope"),
        "first_layer_logit_lens": first("logit_lens"),
        "target_prompt": target_prompt,
    })
