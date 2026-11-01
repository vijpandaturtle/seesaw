"""The computational graph edge-level tools work on: nodes, edges, and where to hook each.

A node either writes to the residual stream (a sender: the embeddings, an
attention head, an MLP) or reads from it (a receiver: a head's query, key or
value input, an MLP's input, the final residual stream the logits come from).
An edge is a sender → receiver pair where the sender comes earlier, so its
output is part of what the receiver reads.

Names, as edge strings use them:
    senders    embed · L5H5 · MLP3
    receivers  L8H6.q · L8H6.k · L8H6.v · MLP3 · logits
    edges      "L5H5->L8H6.v", "MLP0->L9H9.q", "L9H9->logits"

Reading per-head inputs needs TransformerLens's use_split_qkv_input and
use_hook_mlp_in; `split_inputs` switches them on for the duration and
restores the model's settings after, so callers' models come back unchanged.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from transformer_lens import HookedTransformer


@dataclass(frozen=True)
class Sender:
    name: str
    order: int          # position in the forward pass; an edge needs sender.order < receiver.order
    layer: int
    head: int | None    # None for embed and MLPs


@dataclass(frozen=True)
class Receiver:
    name: str
    order: int
    hook: str           # TransformerLens hook whose value is this receiver's input
    head: int | None    # index into the head dimension of a split q/k/v input


def senders(model: HookedTransformer) -> list[Sender]:
    out = [Sender("embed", 0, -1, None)]
    for l in range(model.cfg.n_layers):
        out += [Sender(f"L{l}H{h}", 2 * l + 1, l, h) for h in range(model.cfg.n_heads)]
        if not model.cfg.attn_only:
            out.append(Sender(f"MLP{l}", 2 * l + 2, l, None))
    return out


def receivers(model: HookedTransformer) -> list[Receiver]:
    out = []
    for l in range(model.cfg.n_layers):
        for h in range(model.cfg.n_heads):
            out += [Receiver(f"L{l}H{h}.{x}", 2 * l + 1, f"blocks.{l}.hook_{x}_input", h) for x in "qkv"]
        if not model.cfg.attn_only:
            out.append(Receiver(f"MLP{l}", 2 * l + 2, f"blocks.{l}.hook_mlp_in", None))
    last = model.cfg.n_layers - 1
    out.append(Receiver("logits", 2 * model.cfg.n_layers + 1, f"blocks.{last}.hook_resid_post", None))
    return out


def parse_edge(edge: str, model: HookedTransformer) -> tuple[Sender, Receiver]:
    src, dst = (s.strip() for s in edge.split("->"))
    s = {x.name: x for x in senders(model)}.get(src)
    r = {x.name: x for x in receivers(model)}.get(dst)
    if s is None or r is None:
        raise ValueError(f"unknown edge {edge!r}: senders look like 'embed', 'L5H5', 'MLP3'; "
                         f"receivers like 'L8H6.q', 'MLP3', 'logits'")
    if s.order >= r.order:
        raise ValueError(f"{edge!r}: {src} doesn't come before {dst}, so it can't feed it")
    return s, r



def is_sender_hook(name: str) -> bool:
    return name == "blocks.0.hook_resid_pre" or name.endswith(("hook_z", "hook_mlp_out"))


def sender_output(model: HookedTransformer, cache, s: Sender) -> torch.Tensor:
    """What sender `s` writes to the residual stream: [batch, seq, d_model]."""
    if s.name == "embed":
        return cache["blocks.0.hook_resid_pre"]
    if s.head is not None:
        z = cache[f"blocks.{s.layer}.attn.hook_z"][:, :, s.head, :]           # [b, s, d_head]
        return z @ model.W_O[s.layer, s.head]
    return cache[f"blocks.{s.layer}.hook_mlp_out"]


@contextmanager
def split_inputs(model: HookedTransformer):
    """Expose per-head q/k/v inputs and the MLP input as hook points, then restore the model."""
    before = (model.cfg.use_split_qkv_input, model.cfg.use_hook_mlp_in)
    model.set_use_split_qkv_input(True)
    if not model.cfg.attn_only:
        model.set_use_hook_mlp_in(True)
    try:
        yield
    finally:
        model.set_use_split_qkv_input(before[0])
        if not model.cfg.attn_only:
            model.set_use_hook_mlp_in(before[1])
