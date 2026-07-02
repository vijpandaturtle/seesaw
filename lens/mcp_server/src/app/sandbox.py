import concurrent.futures

from ..models.schemas import ExperimentResult
from ..config import SANDBOX_TIMEOUT


def run_in_sandbox(
    tool_fn,
    tool_kwargs: dict,
    experiment_name: str,
    timeout: int = SANDBOX_TIMEOUT,
) -> ExperimentResult:
    """Run a tool function in an isolated thread with a timeout.

    Catches all exceptions and converts them to a failed ExperimentResult,
    keeping the LangGraph workflow running even when a tool crashes.

    Args:
        tool_fn: The tool function to call.
        tool_kwargs: Keyword arguments to pass to the tool.
        experiment_name: Name for the result (used in error messages).
        timeout: Max seconds to wait before declaring a timeout.

    Returns:
        ExperimentResult with status='success', 'failed', or 'timeout'.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(tool_fn, **tool_kwargs)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return ExperimentResult(
                name=experiment_name,
                tool="unknown",
                model_name="unknown",
                prompts=[],
                status="timeout",
                error=f"Timed out after {timeout}s",
            )
        except Exception as e:
            return ExperimentResult(
                name=experiment_name,
                tool="unknown",
                model_name="unknown",
                prompts=[],
                status="failed",
                error=str(e),
            )
