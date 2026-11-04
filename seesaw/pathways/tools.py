"""Running a pathway's steps through Lens's tool registry.

Every step a pathway takes is a call to a tool in Lens's registry
(lens/mcp_server/src/tools, SEESAW_REGISTRY): the same functions, with the
same arguments, that Lens's own experiments run. Where a step runs follows
Lens:

- given a loaded HookedTransformer, in this process on that model (tests,
  notebooks, a planted copy of a model);
- given a model name, wherever SEESAW_LENS_BACKEND says: in this process on
  Lens's cached copy of the model (local, the default), or on the GPU app in
  lens/modal_app.py as one experiment spec per step (modal).

Only JSON crosses to Modal, so a step can't hand the next one a Python
object. The tools take what rebuilds it instead: the prompts a direction is
fitted on and its layer (steering_vector, directional_ablation), the pairs
LEACE fits on (leace_erasure), the layer whose SAE to load (the feature
tools), the other model's name (model_diff). An edit (ROME, a baked
direction, LEACE erasers) is made, measured and undone inside one call. A
result's data is round-tripped through JSON in-process too, so a pathway
reads exactly the same thing whichever way it ran.
"""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

from seesaw.interp.result import InterpResult


class ToolFailed(RuntimeError):
    """A registry tool failed for a pathway step; the message names the tool and the cause."""


class Tools:
    """Calls registry tools for one pathway run, on one model, saving plots to one folder."""

    def __init__(self, model, output_dir: Path):
        """`model` is a loaded HookedTransformer, or a model name to run by name."""
        self.model = model
        self.output_dir = Path(output_dir)

    @property
    def by_name(self) -> bool:
        return isinstance(self.model, str)

    @property
    def model_name(self) -> str:
        return self.model if self.by_name else self.model.cfg.model_name

    @cached_property
    def n_layers(self) -> int:
        if not self.by_name:
            return self.model.cfg.n_layers
        from transformer_lens.loading_from_pretrained import get_pretrained_model_config

        return get_pretrained_model_config(self.model).n_layers

    def token_lengths(self, prompts: list[str]) -> list[int]:
        """Each prompt's length in tokens (BOS included), without running the model."""
        if not self.by_name:
            return [self.model.to_tokens(p).shape[1] for p in prompts]
        from .catalog import _tokenizer

        tokenizer = _tokenizer(self.model)
        if tokenizer is None:
            raise ToolFailed(f"no tokenizer for {self.model}")
        return [len(tokenizer.encode(p, add_special_tokens=False)) + 1 for p in prompts]

    def __call__(self, tool: str, prompts: list[str], **kwargs) -> InterpResult:
        """Run registry tool `tool` on `prompts` with `kwargs`, and return its data and plots.

        Raises:
            ToolFailed: the tool raised, or the remote run came back failed.
        """
        from lens.mcp_server.src.app.remote import is_remote

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.by_name and is_remote():
            return self._remote(tool, prompts, kwargs)
        return self._local(tool, prompts, kwargs)

    def _local(self, tool: str, prompts: list[str], kwargs: dict) -> InterpResult:
        from lens.mcp_server.src.tools import SEESAW_REGISTRY

        model = self.model
        if self.by_name:
            from lens.mcp_server.src.app.model_session import get_model

            model = get_model(self.model)
        try:
            result = SEESAW_REGISTRY[tool](model=model, prompts=prompts, output_dir=self.output_dir, **kwargs)
        except Exception as exc:                      # noqa: BLE001 — re-raised with the tool's name
            raise ToolFailed(f"{tool}: {type(exc).__name__}: {exc}") from exc
        return InterpResult(data=json.loads(json.dumps(result.data)), plot_paths=list(result.plot_paths))

    def _remote(self, tool: str, prompts: list[str], kwargs: dict) -> InterpResult:
        from lens.mcp_server.src.app.remote import run_experiment_remote

        spec = {"name": tool, "tool": tool, "model_name": self.model, "prompts": prompts,
                "tool_kwargs": kwargs}
        result = run_experiment_remote(spec, output_dir=self.output_dir)
        if result.status != "success":
            raise ToolFailed(f"{tool} on Modal: {result.error}")
        return InterpResult(data=result.data, plot_paths=list(result.plot_paths))
