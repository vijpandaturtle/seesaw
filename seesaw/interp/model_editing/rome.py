"""ROME: rewrite one fact with a rank-one edit to one MLP's output weights.

Meng et al. 2022. An MLP's output matrix acts as a key-value store: the key
is the MLP's hidden activation at the subject's last token, the value is what
it writes to the residual stream. ROME finds

- the key k*, averaged over the edit prompt and a few prefixed versions of it;
- the value v* (as a change δ to the MLP's output) that makes the model say
  the new answer, by gradient descent on δ alone;
- the smallest change to W_out (smallest under the key covariance C) that
  maps k* to the new value: W_out += (C⁻¹k*)ᵀ δ / (C⁻¹k* · k*).

The edit maps k* exactly, so efficacy is close to guaranteed; C decides how
little else changes. ROME estimates C from 100k Wikipedia samples; here it
comes from a short built-in corpus plus shrinkage, which is coarser, so
specificity is measured rather than assumed.

Scored the standard ways (CounterFact): efficacy (the edited prompt now
prefers the new answer), paraphrase (rewordings do too) and specificity
(other subjects with the same relation keep their answers).

Answers: "Can this fact be rewritten at this layer, and what else changes?"
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

from .._helpers import tokens_to_ids
from ..causal_interventions.causal_tracing import subject_span
from ..result import InterpResult

# Generic text the key covariance is estimated from, and the prefixes k* is
# averaged over (ROME samples both from the model; fixed text keeps runs
# reproducible).
CORPUS = [
    "The city council met on Tuesday to discuss the new budget for schools and roads.",
    "She studied physics at university before moving abroad to work as an engineer.",
    "The river flows through three countries before it reaches the sea in the south.",
    "Most of the paintings in the museum were donated by a private collector in 1950.",
    "He played the piano every evening, and on weekends he taught children to sing.",
    "The company announced that its profits had risen sharply during the last quarter.",
    "Scientists have discovered a new species of frog in the forests of the region.",
    "The novel tells the story of a family that survives a long and difficult winter.",
    "After the match, the team's captain thanked the fans for their constant support.",
    "The bridge was built in the nineteenth century and is still used by thousands daily.",
    "Many students prefer to study in the library because it is quiet in the mornings.",
    "The recipe calls for flour, sugar, two eggs and a small amount of fresh butter.",
    "The election results were announced late at night after a long day of counting.",
    "Our neighbours adopted a dog from the shelter, and it follows them everywhere.",
    "The film won several awards, including one for its music and one for its actors.",
    "Heavy rain caused flooding in several towns, and many roads were closed for days.",
]
PREFIXES = ["", "Here is a fact. ", "I read this somewhere. ", "Let me tell you something. ",
            "As everyone knows, "]


def _keys(model: HookedTransformer, layer: int, texts: list[str]) -> torch.Tensor:
    """[n_tokens, d_mlp] MLP hidden activations at `layer` over every token of `texts`."""
    name = f"blocks.{layer}.mlp.hook_post"
    rows = []
    for t in texts:
        with torch.no_grad():
            _, cache = model.run_with_cache(model.to_tokens(t), names_filter=name)
        rows.append(cache[name][0])
    return torch.cat(rows)


def key_covariance(model: HookedTransformer, layer: int, texts: list[str] | None = None,
                   shrinkage: float = 0.1) -> torch.Tensor:
    """Second moment of MLP keys at `layer`, shrunk toward its mean diagonal so it inverts."""
    k = _keys(model, layer, texts or CORPUS).float()
    c = k.T @ k / k.shape[0]
    return (1 - shrinkage) * c + shrinkage * c.diagonal().mean() * torch.eye(c.shape[0])


def _subject_key(model, layer, prompt, subject) -> torch.Tensor:
    name = f"blocks.{layer}.mlp.hook_post"
    keys = []
    for prefix in PREFIXES:
        text = prefix + prompt
        _, end = subject_span(model, text, subject)
        with torch.no_grad():
            _, cache = model.run_with_cache(model.to_tokens(text), names_filter=name)
        keys.append(cache[name][0, end - 1])
    return torch.stack(keys).mean(0).float()


def _delta(model, layer, prompt, subject, target_id, steps: int, lr: float, weight_decay: float):
    """The change to the MLP output at the subject's last token that makes `target_id` likely."""
    _, end = subject_span(model, prompt, subject)
    tokens = model.to_tokens(prompt)
    delta = torch.zeros(model.cfg.d_model, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)

    def add(value, hook):
        value = value.clone()
        value[:, end - 1] = value[:, end - 1] + delta.to(value.dtype)
        return value

    for _ in range(steps):
        opt.zero_grad()
        with torch.enable_grad(), model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_mlp_out", add)]):
            logp = model(tokens)[0, -1].log_softmax(-1)[target_id]
            loss = -logp + weight_decay * delta.norm() ** 2
            # Gradient for delta only: backward() would also fill every weight's .grad.
            (delta.grad,) = torch.autograd.grad(loss, [delta])
        opt.step()
    return delta.detach()


def edit_weights(model: HookedTransformer, layer: int, prompt: str, subject: str, target_new: str,
                 cov: torch.Tensor | None = None, steps: int = 30, lr: float = 0.5,
                 weight_decay: float = 1e-3) -> torch.Tensor:
    """The rank-one update to blocks[layer].mlp.W_out that rewrites the fact. Doesn't apply it."""
    k = _subject_key(model, layer, prompt, subject)
    delta = _delta(model, layer, prompt, subject, tokens_to_ids(model, [target_new])[0],
                   steps, lr, weight_decay)
    c = cov if cov is not None else key_covariance(model, layer)
    u = torch.linalg.solve(c, k)
    return torch.outer(u, delta) / (u @ k)


@contextlib.contextmanager
def applied(model: HookedTransformer, layer: int, update: torch.Tensor):
    """Apply a W_out update inside the block, and put the original weights back after."""
    w = model.blocks[layer].mlp.W_out
    original = w.data.clone()
    w.data += update.to(w.dtype).to(w.device)
    try:
        yield model
    finally:
        w.data.copy_(original)


def _prefers(model, prompts: list[str], better: list[int], worse: list[int]) -> list[bool]:
    out = []
    for p, b, w in zip(prompts, better, worse):
        with torch.no_grad():
            logits = model(model.to_tokens(p))[0, -1]
        out.append(bool(logits[b] > logits[w]))
    return out


def compute(
    model: HookedTransformer,
    prompts: list[str],
    subjects: list[str],
    positive_tokens: list[str],
    target_tokens: list[str],
    layer: int,
    output_dir: Path,
    paraphrase_prompts: list[str] | None = None,
    neighborhood_prompts: list[str] | None = None,
    neighborhood_tokens: list[str] | None = None,
    cov: torch.Tensor | None = None,
) -> InterpResult:
    """Edit each fact at `layer`, one at a time, and score the edit.

    Args:
        model: Loaded HookedTransformer.
        prompts: One prompt per fact, ending right before the answer.
        subjects: The subject inside each prompt.
        positive_tokens: The true answer per fact.
        target_tokens: The new answer to write in per fact.
        layer: The MLP to edit.
        output_dir: Unused (no plot); kept so every tool has the same shape.
        paraphrase_prompts: A rewording of each prompt, same subject and answer.
        neighborhood_prompts: A prompt per fact about a *different* subject
            with the same relation, which the edit shouldn't touch.
        neighborhood_tokens: That prompt's true answer.
        cov: Precomputed key covariance for `layer`.

    Returns:
        InterpResult with efficacy, paraphrase and specificity (fractions of
        facts; specificity over the neighbours right before editing), their
        harmonic mean as `score`, and per-fact results.
    """
    true_ids = tokens_to_ids(model, positive_tokens)
    new_ids = tokens_to_ids(model, target_tokens)
    near_ids = tokens_to_ids(model, neighborhood_tokens) if neighborhood_tokens else None
    c = cov if cov is not None else key_covariance(model, layer)

    # Specificity only counts neighbours the model gets right before any edit:
    # one it already gets wrong can't be broken by the edit.
    near_ok = (_prefers(model, neighborhood_prompts, near_ids, new_ids)
               if neighborhood_prompts and near_ids else [])

    per_fact = []
    for i, (prompt, subject) in enumerate(zip(prompts, subjects)):
        update = edit_weights(model, layer, prompt, subject, target_tokens[i], cov=c)
        with applied(model, layer, update):
            row = {"efficacy": _prefers(model, [prompt], [new_ids[i]], [true_ids[i]])[0]}
            if paraphrase_prompts:
                row["paraphrase"] = _prefers(model, [paraphrase_prompts[i]], [new_ids[i]], [true_ids[i]])[0]
            if near_ok and near_ok[i]:
                row["specificity"] = _prefers(model, [neighborhood_prompts[i]], [near_ids[i]], [new_ids[i]])[0]
        per_fact.append(row)

    keys = ["efficacy"] + [k for k in ("paraphrase", "specificity") if any(k in r for r in per_fact)]
    scores = {}
    for key in keys:
        vals = [r[key] for r in per_fact if key in r]
        scores[key] = sum(vals) / len(vals)
    parts = list(scores.values())
    score = 0.0 if min(parts) == 0 else len(parts) / sum(1 / p for p in parts)
    return InterpResult(
        data={
            "layer":    layer,
            **{k: round(v, 4) for k, v in scores.items()},
            "score":    round(score, 4),
            "per_fact": per_fact,
        },
    )
