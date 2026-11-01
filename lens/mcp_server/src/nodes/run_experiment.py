import inspect
from pathlib import Path

from ..app.model_session import get_model, retire_model
from ..app.remote import is_remote, run_experiment_remote
from ..app.sandbox import run_in_sandbox
from ..config import PLOTS_DIR, SANDBOX_TIMEOUT, SWEEP_TIMEOUT, SWEEP_TOOLS
from ..models.schemas import ExperimentResult
from shared.lens_tools import active_backend

from ..tools import expand_pairings, normalize_tool_kwargs, registry


def run_experiment(state: dict) -> dict:
    """Pop the next experiment off the queue and run it in the sandbox.

    Node 3 in the Lens workflow.

    tool_kwargs come from an LLM reading a prose research plan, so they are
    normalised and checked against the tool's real signature before dispatch.
    A spec that can't satisfy its tool fails here with a message naming the
    missing arguments, rather than as a bare TypeError from inside the call.
    Separate prompt and image lists are expanded to every pairing first.
    """
    queue = list(state["experiment_queue"])
    plots_dir = Path(state["output_dir"]) / "plots" if state.get("output_dir") else PLOTS_DIR
    spec  = queue.pop(0)
    print(f"\n🔬 [run_experiment] '{spec['name']}' using {spec['tool']}")

    prompts, tool_kwargs = expand_pairings(spec["prompts"], spec.get("tool_kwargs") or {})
    if len(prompts) != len(spec["prompts"]):
        print(f"   ↳ {len(spec['prompts'])} prompts × {len(prompts) // len(spec['prompts'])} images "
              f"→ {len(prompts)} pairs")
        spec = dict(spec, prompts=prompts, tool_kwargs=tool_kwargs)

    def failure(error: str) -> ExperimentResult:
        return ExperimentResult(
            name=spec["name"],
            tool=spec["tool"],
            model_name=spec["model_name"],
            prompts=spec["prompts"],
            status="failed",
            error=error,
            tool_kwargs=spec.get("tool_kwargs") or {},
        )

    tool_fn = registry().get(spec["tool"])
    if tool_fn is None:
        result = failure(
            f"Tool '{spec['tool']}' not in the {active_backend()} registry: {list(registry())}"
        )
    elif is_remote() and active_backend() != "seesaw":
        result = failure("The lever backend runs locally only; unset SEESAW_LENS_BACKEND=modal")
    elif is_remote():
        # Modal validates the spec on its side and returns plot bytes, which
        # run_experiment_remote writes into the usual PLOTS_DIR.
        print("   ↗ running on Modal")
        result = run_experiment_remote(spec, output_dir=plots_dir)
    else:
        kwargs, missing, dropped = normalize_tool_kwargs(
            spec["tool"], spec.get("tool_kwargs", {}), len(spec["prompts"])
        )
        if dropped:
            print(f"   ⚠️  {spec['tool']} doesn't accept {dropped} — ignoring")
        if missing:
            result = failure(
                f"Experiment spec for '{spec['tool']}' is missing required "
                f"tool_kwargs {missing}; got {sorted(spec.get('tool_kwargs') or {})}"
            )
        else:
            # The job's folder, not the plan's say: the model never picks where plots go.
            if "output_dir" in inspect.signature(tool_fn).parameters:
                plots_dir.mkdir(parents=True, exist_ok=True)
                kwargs["output_dir"] = plots_dir
            m       = get_model(spec["model_name"])
            timeout = SWEEP_TIMEOUT if spec["tool"] in SWEEP_TOOLS[active_backend()] else SANDBOX_TIMEOUT
            result  = run_in_sandbox(
                tool_fn,
                tool_kwargs={"model": m, "prompts": spec["prompts"], **kwargs},
                experiment_name=spec["name"],
                tool=spec["tool"],
                model_name=spec["model_name"],
                prompts=spec["prompts"],
                timeout=timeout,
            )
            if result.status == "timeout":
                # The thread is still running on `m`, hooks and all; don't let
                # the next experiment share it.
                retire_model(spec["model_name"], m)
                print("   ↳ model retired; the next experiment loads a fresh copy")
            # Tools record the model's own name ("gpt2"), which can differ from
            # the plan's alias ("gpt2-small"). Follow-ups copy this field, so
            # keep the plan's name or get_model loads a second copy.
            result.model_name = spec["model_name"]
            result.tool_kwargs = {k: v for k, v in kwargs.items() if k != "output_dir"}

    icon = "✅" if result.status == "success" else "❌"
    print(f"   {icon} {result.status} | plots={len(result.plot_paths)}")
    if result.status != "success" and result.error:
        print(f"   ↳ {result.error[:300]}")
    return {
        "experiment_queue": queue,
        "last_result": result.__dict__ | {"plot_paths": [str(p) for p in result.plot_paths]},
    }
