"""Turning a library InterpResult into the ExperimentResult Lens's graph and Quill expect."""

from seesaw.interp.result import InterpResult

from ..models.schemas import ExperimentResult


def wrap(result: InterpResult, name: str, tool: str, model, prompts: list[str],
         drop: tuple[str, ...] = ()) -> ExperimentResult:
    """Adapt `result`, leaving out `drop` keys — raw vectors that would only bloat the
    bundle and crowd the numbers out of the interpretation prompt."""
    return ExperimentResult(
        name=name,
        tool=tool,
        model_name=model.cfg.model_name,
        prompts=prompts,
        plot_paths=result.plot_paths,
        data={k: v for k, v in result.data.items() if k not in drop},
        status="success",
    )
