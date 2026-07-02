from ..app.model_session import get_model
from ..app.sandbox import run_in_sandbox
from ..models.schemas import ExperimentResult
from ..tools import TOOL_REGISTRY


def run_experiment(state: dict) -> dict:
    """Pop the next experiment off the queue and run it in the sandbox.

    Node 3 in the Lens workflow.
    """
    queue = list(state["experiment_queue"])
    spec  = queue.pop(0)
    print(f"\n🔬 [run_experiment] '{spec['name']}' using {spec['tool']}")

    tool_fn = TOOL_REGISTRY.get(spec["tool"])
    if tool_fn is None:
        result = ExperimentResult(
            name=spec["name"],
            tool=spec["tool"],
            model_name=spec["model_name"],
            prompts=spec["prompts"],
            status="failed",
            error=f"Tool '{spec['tool']}' not in registry: {list(TOOL_REGISTRY.keys())}",
        )
    else:
        m      = get_model(spec["model_name"])
        result = run_in_sandbox(
            tool_fn,
            tool_kwargs={"model": m, "prompts": spec["prompts"], **spec.get("tool_kwargs", {})},
            experiment_name=spec["name"],
        )

    icon = "✅" if result.status == "success" else "❌"
    print(f"   {icon} {result.status} | plots={len(result.plot_paths)}")
    return {
        "experiment_queue": queue,
        "last_result": result.__dict__ | {"plot_paths": [str(p) for p in result.plot_paths]},
    }
