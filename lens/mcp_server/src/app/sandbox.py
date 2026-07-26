import concurrent.futures
import traceback

from ..models.schemas import ExperimentResult
from ..config import SANDBOX_TIMEOUT


def run_in_sandbox(
    tool_fn,
    tool_kwargs: dict,
    experiment_name: str,
    tool: str = "unknown",
    model_name: str = "unknown",
    prompts: list[str] | None = None,
    timeout: int = SANDBOX_TIMEOUT,
) -> ExperimentResult:
    """Run a tool function in a worker thread with a timeout.

    Catches all exceptions and converts them to a failed ExperimentResult,
    keeping the LangGraph workflow running even when a tool crashes. The
    caller passes the spec's identity so a failure is still attributable to
    the tool and model that produced it.

    Timeouts stop *waiting* on the tool; they cannot stop the tool. Python
    can't kill a running thread, so a wedged experiment keeps holding its
    memory and its share of the GPU until it returns on its own. Bounding
    that needs process isolation (the tool would have to load the model in
    the child rather than receive it), which is the change to make before
    Lens runs anywhere metered.

    Args:
        tool_fn: The tool function to call.
        tool_kwargs: Keyword arguments to pass to the tool.
        experiment_name: Name for the result (used in error messages).
        tool: Registry name of the tool, recorded on failure results.
        model_name: Model the experiment targeted, recorded on failure results.
        prompts: Prompts the experiment ran on, recorded on failure results.
        timeout: Max seconds to wait before declaring a timeout.

    Returns:
        ExperimentResult with status='success', 'failed', or 'timeout'.
    """
    def failed(status: str, error: str) -> ExperimentResult:
        return ExperimentResult(
            name=experiment_name,
            tool=tool,
            model_name=model_name,
            prompts=list(prompts or []),
            status=status,
            error=error,
        )

    # Not a context manager: __exit__ would shutdown(wait=True) and block on
    # the very thread the timeout is meant to walk away from.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(tool_fn, **tool_kwargs)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return failed("timeout", f"Timed out after {timeout}s (thread still running)")
        except Exception as e:
            return failed("failed", f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")
    finally:
        executor.shutdown(wait=False)
