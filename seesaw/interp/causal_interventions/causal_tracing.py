"""Causal tracing: where does the model retrieve a fact about the subject?

Meng et al. 2022 (ROME). Corrupt the subject by adding noise to its token
embeddings, which knocks the answer's probability down. Then restore one
piece of the clean run, at one layer and one group of positions, and see how
much of the probability comes back. Three kinds of piece are restored:

- the residual stream (everything computed so far),
- the MLP output, over a window of layers centred on the layer,
- the attention output, over the same window.

Positions are grouped so prompts of different lengths average together:
the subject's tokens before its last, the subject's last token, the tokens
between the subject and the end, and the final token.

The ROME pattern: MLPs at the subject's last token matter in early-middle
layers (the fact is recalled there), and attention at the final token
matters later (it is moved to the answer).

Answers: "At which layer and position is this fact retrieved?"
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from transformer_lens import HookedTransformer

matplotlib.use("Agg")

from .._helpers import tokens_to_ids
from ..result import InterpResult

GROUPS = ("subject_first", "subject_last", "further", "final")
KINDS = ("resid", "mlp", "attn")
_HOOK = {"resid": "hook_resid_post", "mlp": "hook_mlp_out", "attn": "hook_attn_out"}


def subject_span(model: HookedTransformer, prompt: str, subject: str) -> tuple[int, int]:
    """Token positions [start, end) of `subject` inside `prompt`, counting the BOS token."""
    at = prompt.find(subject)
    if at < 0:
        raise ValueError(f"subject {subject!r} is not in prompt {prompt!r}")
    start = model.to_tokens(prompt[:at]).shape[1] if at else 1
    end = model.to_tokens(prompt[:at + len(subject)]).shape[1]
    return start, end


def _groups(model, prompt: str, subject: str) -> dict[str, list[int]]:
    start, end = subject_span(model, prompt, subject)
    final = model.to_tokens(prompt).shape[1] - 1
    return {
        "subject_first": list(range(start, end - 1)),
        "subject_last": [end - 1],
        "further": list(range(end, final)),
        "final": [final],
    }


def compute(
    model: HookedTransformer,
    prompts: list[str],
    subjects: list[str],
    positive_tokens: list[str],
    output_dir: Path,
    noise: float = 3.0,
    n_noise: int = 5,
    window: int = 3,
    seed: int = 0,
) -> InterpResult:
    """Trace each prompt's fact, restoring each kind of piece at each layer and position group.

    Args:
        model: Loaded HookedTransformer.
        prompts: Factual prompts, each ending right before the answer.
        subjects: The subject inside each prompt, e.g. "The Eiffel Tower".
        positive_tokens: The true answer token per prompt, e.g. " Paris".
        output_dir: Directory to save the plot.
        noise: Noise added to subject embeddings, in multiples of the
            embedding matrix's standard deviation (ROME uses 3).
        n_noise: Noise samples per prompt, averaged.
        window: Layers of MLP or attention output restored together, centred
            on the layer (a single layer's MLP restores little on its own).
        seed: Seed for the noise.

    Returns:
        InterpResult with [kind][layer][group] recovery (0 = still corrupted,
        1 = fully clean), the clean and corrupted answer probabilities, and
        each kind's peak.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    answers = tokens_to_ids(model, positive_tokens)
    n_layers = model.cfg.n_layers
    sigma = noise * model.W_E.std().item()
    gen = torch.Generator().manual_seed(seed)
    half = window // 2

    # Restored probabilities are summed per cell and normalised once at the end
    # (as ROME averages): a per-prompt ratio blows up on prompts the noise barely moves.
    sums = {k: torch.zeros(n_layers, len(GROUPS)) for k in KINDS}
    clean_sum, corrupt_sum = torch.zeros(len(GROUPS)), torch.zeros(len(GROUPS))
    clean_ps, corrupt_ps = [], []

    for prompt, subject, answer in zip(prompts, subjects, answers):
        groups = _groups(model, prompt, subject)
        tokens = model.to_tokens(prompt).repeat(n_noise, 1)
        start, end = subject_span(model, prompt, subject)
        eps = torch.randn(n_noise, end - start, model.cfg.d_model, generator=gen).to(
            model.cfg.device, model.cfg.dtype) * sigma

        def corrupt(value, hook):
            value = value.clone()
            value[:, start:end] += eps
            return value

        def prob(extra=()) -> float:
            with torch.no_grad(), model.hooks(fwd_hooks=[("hook_embed", corrupt), *extra]):
                logits = model(tokens)
            return logits[:, -1].softmax(-1)[:, answer].mean().item()

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens[:1], names_filter=lambda n: any(
                n.endswith(h) for h in _HOOK.values()))
            p_clean = model(tokens[:1])[0, -1].softmax(-1)[answer].item()
        p_corrupt = prob()
        clean_ps.append(p_clean)
        corrupt_ps.append(p_corrupt)

        for g, name in enumerate(GROUPS):
            positions = groups[name]
            if not positions:
                continue
            clean_sum[g] += p_clean
            corrupt_sum[g] += p_corrupt
            for kind in KINDS:
                for layer in range(n_layers):
                    layers = [layer] if kind == "resid" else range(
                        max(0, layer - half), min(n_layers, layer + half + 1))
                    hooks = []
                    for l in layers:
                        name_l = f"blocks.{l}.{_HOOK[kind]}"
                        clean = cache[name_l][0, positions]

                        def restore(value, hook, clean=clean, positions=positions):
                            value = value.clone()
                            value[:, positions] = clean
                            return value
                        hooks.append((name_l, restore))
                    sums[kind][layer, g] += prob(hooks)

    gap = clean_sum - corrupt_sum
    safe = torch.where(gap.abs() > 1e-6, gap, torch.ones_like(gap))
    effects = {k: torch.where(gap.abs() > 1e-6, (sums[k] - corrupt_sum) / safe,
                              torch.zeros_like(gap)).tolist() for k in KINDS}
    peaks = {}
    for kind in KINDS:
        grid = torch.tensor(effects[kind])
        layer, g = divmod(int(grid.argmax()), len(GROUPS))
        peaks[kind] = {"layer": layer, "group": GROUPS[g], "recovery": round(grid.max().item(), 4)}

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, kind in zip(axes, KINDS):
        im = ax.imshow(torch.tensor(effects[kind]).T, aspect="auto", cmap="Purples", vmin=0, vmax=1)
        ax.set_title({"resid": "Residual stream", "mlp": f"MLP (window {window})",
                      "attn": f"Attention (window {window})"}[kind])
        ax.set_xlabel("Layer")
        ax.set_yticks(range(len(GROUPS)))
        ax.set_yticklabels(GROUPS)
    fig.colorbar(im, ax=axes, label="recovery")
    plot_path = output_dir / "causal_tracing.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return InterpResult(
        plot_paths=[plot_path],
        data={
            "effects":         {k: [[round(v, 4) for v in row] for row in effects[k]] for k in KINDS},
            "groups":          list(GROUPS),
            "peaks":           peaks,
            "clean_prob":      round(sum(clean_ps) / len(clean_ps), 4),
            "corrupted_prob":  round(sum(corrupt_ps) / len(corrupt_ps), 4),
            "window":          window,
            "noise":           noise,
        },
    )
