"""Loading sparse autoencoders, and reading where they attach and what Neuronpedia calls them.

Seesaw doesn't train SAEs; it loads published ones through SAELens (Gemma
Scope, Llama Scope, the GPT-2 residual SAEs, ...). Every feature tool takes
the loaded SAE object and asks it two things through these helpers: which
hook point it reads, and its Neuronpedia source id, so each feature can be
linked to its dashboard and explanation.

Anything with `.encode()`, `.W_dec` and `.cfg.metadata.hook_name` works, so a
hand-built SAE (or a test double) can stand in for a SAELens one.
"""

from __future__ import annotations

import re

_SAE_CACHE: dict[tuple[str, str, str], object] = {}

# TransformerLens model names that SAELens's directory files under another name.
_MODEL_ALIASES = {"gpt2": "gpt2-small"}

# The residual-stream SAE family to use per model when the caller doesn't
# name one: the release Neuronpedia hosts, so features come back labelled.
PREFERRED_RELEASES = {
    "gpt2-small":          "gpt2-small-res-jb",
    "pythia-70m-deduped":  "pythia-70m-deduped-res-sm",
    "gemma-2-2b":          "gemma-scope-2b-pt-res-canonical",
}


def find_sae(model_name: str, layer: int, release: str | None = None) -> tuple[str, str, str | None]:
    """Pick a published residual-stream SAE for `model_name` at `layer`, without downloading it.

    Args:
        model_name: TransformerLens model name ('gpt2', 'gemma-2-2b', ...).
        layer: Block index the SAE should read.
        release: A specific SAELens release; default PREFERRED_RELEASES, then
            any release for this model with "res" in its name.

    Returns:
        (release, sae_id, neuronpedia_source) — pass the first two to load_sae.
        neuronpedia_source is None when Neuronpedia doesn't host that SAE.

    Raises:
        LookupError: no residual SAE for this model and layer in SAELens's directory.
    """
    from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

    directory = get_pretrained_saes_directory()
    name = _MODEL_ALIASES.get(model_name, model_name)
    candidates = [release] if release else (
        [PREFERRED_RELEASES[name]] if name in PREFERRED_RELEASES else []
    ) + sorted(r for r, e in directory.items() if e.model == name and "res" in r)

    pattern = re.compile(rf"(?:blocks\.|layer_){layer}(?:[./]|$)")
    for rel in dict.fromkeys(candidates):
        entry = directory.get(rel)
        if entry is None:
            continue
        for sae_id in entry.saes_map:
            if pattern.search(sae_id):
                return rel, sae_id, (entry.neuronpedia_id or {}).get(sae_id)
    raise LookupError(
        f"no residual-stream SAE for {model_name!r} at layer {layer} in SAELens's directory "
        f"(tried {list(dict.fromkeys(candidates))[:5]})"
    )


def load_sae(release: str, sae_id: str, device: str = "cpu"):
    """Load (and cache) a pretrained SAE, e.g. ("gpt2-small-res-jb", "blocks.8.hook_resid_pre").

    The release/sae_id names are SAELens's; `sae_lens` lists them in its
    pretrained SAE directory.
    """
    key = (release, sae_id, device)
    if key not in _SAE_CACHE:
        from sae_lens import SAE

        _SAE_CACHE[key] = SAE.from_pretrained(release, sae_id, device=device)
    return _SAE_CACHE[key]


def hook_name(sae) -> str:
    """The TransformerLens hook point the SAE was trained on."""
    return sae.cfg.metadata.hook_name


def neuronpedia_source(sae) -> str | None:
    """'gpt2-small/8-res-jb'-style id, or None when Neuronpedia doesn't host this SAE."""
    metadata = getattr(sae.cfg, "metadata", None)
    return getattr(metadata, "neuronpedia_id", None) if metadata is not None else None
