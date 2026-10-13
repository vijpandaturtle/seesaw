"""MCP tool registration for Lens.

The five original experiments (logit lens, attention patterns, ablation,
activation patching, direct logit attribution) are each their own MCP tool,
with typed arguments. Every other Lens tool — patching on a shortlist,
circuit evaluation, probes, steering, SAE features — goes through `run_tool`,
which takes the tool's name and its arguments and validates them exactly as
the workflow does. Either way a client can call experiments one at a time
rather than going through the full research-plan workflow. The orchestrator uses the workflow graph
directly for the automated pipeline; these tools are for ad-hoc / external
use (e.g. calling Lens from Claude Desktop or another agent).
"""

import json
from dataclasses import asdict

from fastmcp import FastMCP

from ..app.model_session import get_model
from ..app.sandbox import run_in_sandbox
from ..tools import (
    TOOL_REGISTRY,
    normalize_tool_kwargs,
    run_ablation,
    run_activation_patching,
    run_attention_pattern,
    run_direct_logit_attribution,
    run_logit_lens,
)


def _to_json(result) -> str:
    d = asdict(result)
    d["plot_paths"] = [str(p) for p in d["plot_paths"]]
    return json.dumps(d, default=str)


def register_mcp_tools(mcp: FastMCP) -> None:
    @mcp.tool()
    def logit_lens(model_name: str, prompts: list[str], pos: int = -1, top_k: int = 5) -> str:
        """Project the residual stream at each layer to vocabulary space.

        Shows how the model's prediction evolves layer by layer — answers
        "where does the model decide on the answer?"

        Args:
            model_name: TransformerLens model name (e.g. 'gpt2', 'gpt2-small').
            prompts: Input prompt strings to run through the model.
            pos: Token position to inspect (-1 = last token).
            top_k: Number of top tokens to track per layer.

        Returns:
            JSON ExperimentResult with a heatmap plot path and layer-by-layer
            top-token data.
        """
        model  = get_model(model_name)
        result = run_in_sandbox(
            run_logit_lens,
            tool_kwargs={"model": model, "prompts": prompts, "pos": pos, "top_k": top_k},
            experiment_name="Logit Lens",
        )
        return _to_json(result)

    @mcp.tool()
    def attention_pattern(model_name: str, prompts: list[str], layers: list[int] | None = None) -> str:
        """Visualise attention weights per head per layer.

        Answers "what does each head attend to?" Generates candidate heads
        for ablation — attention alone does not prove causal importance.

        Args:
            model_name: TransformerLens model name.
            prompts: Input prompt strings.
            layers: Specific layers to visualise. Defaults to 5 evenly-spaced layers.

        Returns:
            JSON ExperimentResult with one plot path per layer and per-head
            attention data.
        """
        model  = get_model(model_name)
        result = run_in_sandbox(
            run_attention_pattern,
            tool_kwargs={"model": model, "prompts": prompts, "layers": layers},
            experiment_name="Attention Pattern",
        )
        return _to_json(result)

    @mcp.tool()
    def ablation(
        model_name: str,
        prompts: list[str],
        positive_tokens: list[str],
        negative_tokens: list[str],
        ablation_type: str = "mean",
    ) -> str:
        """Silence each attention head and measure the drop in logit difference.

        Answers "is this component causally necessary for the behaviour?" A
        large drop means the head matters; no drop means it's redundant or
        irrelevant for this task.

        Args:
            model_name: TransformerLens model name.
            prompts: Input prompt strings.
            positive_tokens: Tokens to boost (e.g. [" she"] for gender bias, [" Mary"] for IOI).
            negative_tokens: Tokens to suppress (e.g. [" he"], [" John"]).
            ablation_type: "zero" or "mean" ablation. Mean is generally preferred —
                zero pushes the model out of distribution.

        Returns:
            JSON ExperimentResult with an ablation heatmap and top important heads.
        """
        model  = get_model(model_name)
        result = run_in_sandbox(
            run_ablation,
            tool_kwargs={
                "model": model,
                "prompts": prompts,
                "positive_tokens": positive_tokens,
                "negative_tokens": negative_tokens,
                "ablation_type": ablation_type,
            },
            experiment_name=f"Ablation ({ablation_type})",
        )
        return _to_json(result)

    @mcp.tool()
    def activation_patching(
        model_name: str,
        prompts: list[str],
        corrupted_prompts: list[str],
        positive_tokens: list[str],
        negative_tokens: list[str],
    ) -> str:
        """Patch clean activations into a corrupted run and measure recovery.

        Answers "which layer and token position causally carries the signal?"
        Recovery near 1.0 at a (layer, position) means that's where the
        critical information lives.

        Args:
            model_name: TransformerLens model name.
            prompts: Clean prompts (model gets the right answer).
            corrupted_prompts: Corrupted prompts (e.g. names swapped, neutral subject).
            positive_tokens: Tokens to boost.
            negative_tokens: Tokens to suppress.

        Returns:
            JSON ExperimentResult with a layer x position recovery heatmap.
        """
        model  = get_model(model_name)
        result = run_in_sandbox(
            run_activation_patching,
            tool_kwargs={
                "model": model,
                "prompts": prompts,
                "corrupted_prompts": corrupted_prompts,
                "positive_tokens": positive_tokens,
                "negative_tokens": negative_tokens,
            },
            experiment_name="Activation Patching",
        )
        return _to_json(result)

    @mcp.tool()
    def direct_logit_attribution(
        model_name: str,
        prompts: list[str],
        positive_tokens: list[str],
        negative_tokens: list[str],
        pos: int = -1,
    ) -> str:
        """Decompose the final logit difference into per-head and per-MLP contributions.

        Answers "which components write the prediction into the residual
        stream, and in which direction?" Positive attribution promotes the
        positive token; negative attribution suppresses it.

        Args:
            model_name: TransformerLens model name.
            prompts: Input prompt strings.
            positive_tokens: Tokens to boost.
            negative_tokens: Tokens to suppress.
            pos: Token position to read logits from (-1 = last token).

        Returns:
            JSON ExperimentResult with head/MLP attribution heatmaps and top heads.
        """
        model  = get_model(model_name)
        result = run_in_sandbox(
            run_direct_logit_attribution,
            tool_kwargs={
                "model": model,
                "prompts": prompts,
                "positive_tokens": positive_tokens,
                "negative_tokens": negative_tokens,
                "pos": pos,
            },
            experiment_name="Direct Logit Attribution",
        )
        return _to_json(result)

    @mcp.tool()
    def run_tool(model_name: str, tool: str, prompts: list[str], tool_kwargs: dict | None = None) -> str:
        """Run any Lens tool by name. The tool_selection_guide prompt lists them.

        Use this for the tools without their own MCP entry: attribution_patching,
        head_patching, circuit_evaluation, edge_attribution_patching,
        path_patching, circuit_tracing, counterfactual, linear_probe,
        steering_vector, directional_ablation, feature_search,
        feature_attribution, feature_clamping.

        Args:
            model_name: TransformerLens model name (e.g. 'gpt2').
            tool: Tool name, e.g. 'head_patching'.
            prompts: Input prompts.
            tool_kwargs: The tool's other arguments, e.g. {"corrupted_prompts": [...],
                "positive_tokens": [" Mary"], "negative_tokens": [" John"], "heads": ["L9H9"]}.

        Returns:
            JSON ExperimentResult, or a JSON error naming missing arguments.
        """
        if tool not in TOOL_REGISTRY:
            return json.dumps({"status": "failed", "error": f"unknown tool {tool!r}; "
                               f"available: {sorted(TOOL_REGISTRY)}"})
        kwargs, missing, dropped = normalize_tool_kwargs(tool, tool_kwargs or {}, len(prompts))
        if missing:
            return json.dumps({"status": "failed", "error": f"{tool} needs {missing}",
                               "ignored": dropped})
        result = run_in_sandbox(
            TOOL_REGISTRY[tool],
            tool_kwargs={"model": get_model(model_name), "prompts": prompts, **kwargs},
            experiment_name=tool, tool=tool, model_name=model_name, prompts=prompts,
        )
        return _to_json(result)
