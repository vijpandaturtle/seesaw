"""LEACE: erase a binary concept from the residual stream, with the least change that makes it unreadable.

Belrose et al. 2023. Given activations X and a binary label z, LEACE finds
the affine edit x -> x - P(x - μ) that leaves no linear classifier able to
predict z any better than chance, while moving x as little as possible
(measured after whitening). For a binary concept P is rank one:

    W = Σ^(-1/2)            (whitening, from the activations' covariance)
    u = W σ / |W σ|         (σ = covariance of x with z)
    P = W⁺ u uᵀ W

Unlike projecting out the difference of means, it accounts for how the
activations are spread, so it removes the concept along the direction a
classifier would actually use.

With a few prompts in many dimensions, Σ is far from full rank, so it is
shrunk toward a multiple of the identity before inverting. One eraser is fit
per layer, from clean activations, and they're applied at every layer
together: erasing at one layer alone lets later layers read the concept off
the tokens again.

Answers: "If no linear readout could see this concept, would the behaviour change?"
"""

from __future__ import annotations

import contextlib

import torch
from transformer_lens import HookedTransformer

from ..metrics import last_positions


def fit(x: torch.Tensor, z: torch.Tensor, shrinkage: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """The eraser for activations x [n, d] and binary labels z [n]: (mean, P) with x' = x - (x - mean) Pᵀ."""
    x = x.float()
    z = z.float()
    mu = x.mean(0)
    xc = x - mu
    sigma = xc.T @ xc / x.shape[0]
    sigma = (1 - shrinkage) * sigma + shrinkage * sigma.diagonal().mean() * torch.eye(x.shape[1])
    cross = xc.T @ (z - z.mean()) / x.shape[0]                 # [d]
    vals, vecs = torch.linalg.eigh(sigma)
    vals = vals.clamp(min=1e-8)
    w = vecs @ torch.diag(vals.rsqrt()) @ vecs.T               # Σ^(-1/2)
    w_inv = vecs @ torch.diag(vals.sqrt()) @ vecs.T            # Σ^(1/2)
    u = w @ cross
    u = u / u.norm().clamp(min=1e-12)
    return mu, w_inv @ torch.outer(u, u) @ w


def _sites(model: HookedTransformer) -> list[str]:
    names = [f"blocks.{l}.hook_resid_pre" for l in range(model.cfg.n_layers)]
    return names + [f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"]


def fit_all(model: HookedTransformer, prompts_a: list[str], prompts_b: list[str],
            shrinkage: float = 0.5) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """One eraser per residual-stream site, fit at each prompt's last token (A labelled 1, B 0)."""
    sites = _sites(model)
    rows = []
    for prompts in (prompts_a, prompts_b):
        tokens = model.to_tokens(prompts)
        pos = last_positions(model, prompts).to(tokens.device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in sites)
        r = torch.arange(len(prompts), device=tokens.device)
        rows.append({s: cache[s][r, pos].float().cpu() for s in sites})
    z = torch.tensor([1.0] * len(prompts_a) + [0.0] * len(prompts_b))
    return {s: fit(torch.cat([rows[0][s], rows[1][s]]), z, shrinkage) for s in sites}


@contextlib.contextmanager
def erased(model: HookedTransformer, erasers: dict[str, tuple[torch.Tensor, torch.Tensor]]):
    """Run the model with every eraser applied at its site, at every position."""
    dev, dt = model.cfg.device, model.cfg.dtype

    def make(mu, p):
        mu, p = mu.to(dev, dt), p.to(dev, dt)
        return lambda value, hook: value - (value - mu) @ p.T

    with model.hooks(fwd_hooks=[(s, make(mu, p)) for s, (mu, p) in erasers.items()]):
        yield model


def random_erasers(erasers: dict, seed: int = 0) -> dict:
    """Same-rank erasers along random directions: the control for `erased`."""
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for s, (mu, p) in erasers.items():
        r = torch.randn(p.shape[0], generator=gen)
        r = r / r.norm()
        out[s] = (mu, torch.outer(r, r))
    return out
