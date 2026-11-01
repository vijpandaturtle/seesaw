"""The plain result shape every `seesaw.interp` function returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class InterpResult:
    """Raw output of one interpretability computation — no agent framing.

    Just the numbers (`data`) and any plots saved along the way. No `tool`,
    `name`, `status`, or `summary` — those are agent-specific concerns that
    `lens/mcp_server/src/tools/` adds when it wraps this into an
    `ExperimentResult` for the agent loop.
    """

    data: dict
    plot_paths: list[Path] = field(default_factory=list)
