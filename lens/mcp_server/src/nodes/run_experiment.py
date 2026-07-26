from ..app.model_session import get_model
from ..app.sandbox import run_in_sandbox
from ..models.schemas import ExperimentResult
from ..tools import TOOL_REGISTRY, normalize_tool_kwargs


def run_experiment(state: dict) -> dict:
    """Pop the next experiment off the queue and run it in the sandbox.

    Node 3 in the Lens workflow.

    tool_kwargs come from an LLM reading a prose research plan, so they are
    normalised and checked against the tool's real signature before dispatch.
    A spec that can't satisfy its tool fails here with a message naming the
    missing arguments, rather than as a bare TypeError from inside the call.
    """
    queue = list(state["experiment_queue"])
    spec  = queue.pop(0)
    print(f"\n🔬 [run_experiment] '{spec['name']}' using {spec['tool']}")

    def failure(error: str) -> ExperimentResult:
        return ExperimentResult(
            name=spec["name"],
            tool=spec["tool"],
            model_name=spec["model_name"],
            prompts=spec["prompts"],
            status="failed",
            error=error,
        )

    tool_fn = TOOL_REGISTRY.get(spec["tool"])
    if tool_fn is None:
        result = failure(
            f"Tool '{spec['tool']}' not in registry: {list(TOOL_REGISTRY.keys())}"
        )
    else:
        kwargs, missing = normalize_tool_kwargs(
            spec["tool"], spec.get("tool_kwargs", {}), len(spec["prompts"])
        )
        if missing:
            result = failure(
                f"Experiment spec for '{spec['tool']}' is missing required "
                f"tool_kwargs {missing}; got {sorted(spec.get('tool_kwargs') or {})}"
            )
        else:
            m      = get_model(spec["model_name"])
            result = run_in_sandbox(
                tool_fn,
                tool_kwargs={"model": m, "prompts": spec["prompts"], **kwargs},
                experiment_name=spec["name"],
                tool=spec["tool"],
                model_name=spec["model_name"],
                prompts=spec["prompts"],
            )

    icon = "✅" if result.status == "success" else "❌"
    print(f"   {icon} {result.status} | plots={len(result.plot_paths)}")
    if result.status != "success" and result.error:
        print(f"   ↳ {result.error[:300]}")
    return {
        "experiment_queue": queue,
        "last_result": result.__dict__ | {"plot_paths": [str(p) for p in result.plot_paths]},
    }
