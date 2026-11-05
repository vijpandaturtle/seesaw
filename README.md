<div align="center">

# Seesaw

**A multi-agent framework for automated mechanistic interpretability research**

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Built with LangGraph](https://img.shields.io/badge/built%20with-LangGraph-1c1c1c.svg)](https://github.com/langchain-ai/langgraph)
[![MCP](https://img.shields.io/badge/protocol-MCP-purple.svg)](https://modelcontextprotocol.io)

[Setup & Usage Docs](DOCS.md) · [Use-cases](docs/use-cases.md) · [Pathways](#pathways) · [Architecture](#architecture) · [Blog](https://mindsandai.substack.com)

</div>

<br>

<p align="center">
  <img src="assets/seesaw-v0.jpg" alt="Seesaw v0 Architecture" width="720">
</p>

<br>

## Overview

Seesaw automates the hypothesis → experiment → critique loop in mechanistic interpretability. You ask a research question; three agents handle the rest.

<div align="center">

**Scout** `question → research plan` → **Lens** `plan → results bundle` → **Quill** `bundle → critique report`

</div>

Most questions don't need a free-form plan. Seesaw ships eight **pathways**: fixed, hand-coded investigations that end in a verdict, each checked against a known answer on GPT-2 small. Scout picks the pathway that fits your question and chooses its inputs. You approve both before anything runs on a model. Lens runs the pathway, and Quill critiques the result.

Every result is labelled with how strong its evidence is (correlational, estimate, or causal) and whether its controls ran. Each agent can also run on its own as an [MCP](https://modelcontextprotocol.io) server; see [DOCS.md](DOCS.md).

<br>

## Quick Start

```bash
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[experiments]"   # Lens needs torch + TransformerLens

cp .env.example .env                 # set ANTHROPIC_API_KEY
```

Then ask a question whose answer is already known, so you can check the verdict:

```bash
seesaw jobs submit "Does GPT-2 use the gender a job title implies to choose 'she' or 'he'?" \
  --pathway concept_direction --follow
```

`--follow` streams the job's log and asks you to approve at each gate: once for Scout's plan and inputs, once before Quill critiques the results. The job runs on your machine and is stored in a local SQLite file. Artifacts land in `outputs/jobs/<id>/`.

Leave out `--pathway` and Scout chooses one (or none, and plans the experiments itself). Add `--search` to let Scout search arXiv and the web while it plans; this needs `FIRECRAWL_API_KEY` for web search.

**[docs/use-cases.md](docs/use-cases.md)** lists sample questions for each pathway, with the ★ known-answer ones to start with. Full setup, the CLI reference, and deployment notes are in **[DOCS.md](DOCS.md)**.

<br>

## The `seesaw` command

| Command | Does |
|---|---|
| `seesaw jobs submit "<question>"` | Start a research job: Scout → gate → Lens → gate → Quill. `--pathway`, `--model`, `--search`, `--auto-approve`, `--follow` |
| `seesaw jobs list` / `show` / `logs` | See jobs, their status and artifacts, and the worker's log |
| `seesaw jobs approve` / `retry` / `cancel` | Drive a job: clear a gate, re-run a failed stage, stop it |
| `seesaw jobs rerun <id> [--model M]` | Run a job's pathway again on its approved inputs, from Lens, optionally on another model |
| `seesaw run "<question>"` | The whole pipeline in your terminal, with no job record and no pathway |
| `seesaw plan "<question>"` · `seesaw critique <bundle.json>` | Scout only, or Quill only |
| `seesaw eval <task>` · `seesaw fixtures` | The eval suite, and the meta-eval that checks its graders |

The dashboard ([seesaw-web](https://github.com/vijpandaturtle/seesaw-web)) reads the same job store, so a job started in one shows in the other.

<br>

## Pathways

| Purpose | Pathway | What it does |
|---|---|---|
| Understand a capability | `circuit_discovery` | Nominates heads, verifies heads and MLPs by patching, and checks the circuit is faithful |
| Find and steer a concept | `concept_direction` | Fits a concept direction on half the pairs, then removes and steers it on the other half against random directions |
| Build a detector | `probe_monitor` | Trains probes, picks a layer, and tests on held-out texts against shuffled labels, an embeddings-only probe, and bag-of-words |
| Audit a spurious dependence | `spurious_audit` | Measures how much an attribute moves the answer, then erases it with LEACE and checks the task survives |
| Locate a fact | `factual_recall` | Causal tracing over layers and positions |
| Edit a fact | `editing_comparison` | ROME edits at every layer, compared with the traced layer |
| Diff two models | `model_diffing` | Per-prompt KL, which tokens the change favours, and which component's weights account for it |
| Explain an output | `feature_explain` | Sweeps GPT-2 small SAEs for the features that drive the answer (gpt2 only) |

The pathways are deterministic, and their random baselines are seeded. A rerun of a job gives the same verdict on the same model and inputs. Their inputs are defined in [`seesaw/pathways/catalog.py`](seesaw/pathways/catalog.py), and the known-answer checks are in [`tests/test_pathways_gpt2.py`](tests/test_pathways_gpt2.py).

<br>

## Agents

<table>
<tr>
<td width="33%" valign="top">

### 🔭 Scout
**The Research Planner**

Turns a question into a research plan: it chooses a pathway and its inputs, or plans free-form experiments when no pathway fits. The plan includes hypotheses, the target model, and extra experiments. Literature search over arXiv and the web is opt-in (`--search`).

**Tools**
`search_arxiv` · `search_arxiv_web` · `search_web` · `scrape_url` · `save_research_plan`

**Stack**
LangGraph ReAct · Claude · Firecrawl

</td>
<td width="33%" valign="top">

### 🔬 Lens
**The Experiment Runner**

Runs the plan: the pathway first, then any extra experiments, then an LLM interpretation of each result. Lens has 18 tools for text models, grouped by the evidence they give:

| Kind | Tools |
|---|---|
| Descriptive | `logit_lens` · `attention_pattern` · `direct_logit_attribution` · `linear_probe` · `feature_search` · `counterfactual` |
| Estimate (nominates) | `attribution_patching` · `feature_attribution` · `edge_attribution_patching` · `circuit_tracing` |
| Causal (verifies) | `ablation` · `activation_patching` · `head_patching` · `circuit_evaluation` · `path_patching` · `steering_vector` · `directional_ablation` · `feature_clamping` |

With `SEESAW_INTERP_BACKEND=lever`, Lens runs six tools on vision-language models instead (LLaVA, Qwen-VL, CLIP). Full catalog: [`shared/lens_tools.py`](shared/lens_tools.py).

**Stack**
LangGraph StateGraph · Claude · TransformerLens · SAELens · Neuronpedia

</td>
<td width="33%" valign="top">

### 🪶 Quill
**The Reviewer**

Reviews the results bundle like a peer reviewer, treating the pathway's verdict as the primary evidence. It flags methodological issues, unsupported conclusions, and missing experiments, and writes follow-up specs Lens can run directly.

**Stack**
LangGraph StateGraph · Claude Opus

</td>
</tr>
</table>

<br>

## Architecture

Every agent shares the same internal structure — `app/`, `config/`, `models/`, `tools/`, `routers/`, `resources/`, `prompts/`, `server.py` — so the codebase reads consistently whether you're in Scout, Lens, or Quill. Details in [DOCS.md](DOCS.md#directory-structure).

The pipeline is deliberately simple: a fixed-order **prompt-chaining workflow**, not autonomous multi-agent routing. Scout always runs first, Lens second, Quill third, and no agent decides who runs next. That keeps the system predictable and easy to debug.

The interpretability code is a plain library, separate from the agents: `seesaw.interp` holds the tools and `seesaw.pathways` holds the pathways. You can call either from a notebook without any agent; see [DOCS.md](DOCS.md#library-api).

<br>

## Motivation

Most mechanistic interpretability work is done by hand — identify a circuit, ablate it, write it up. Seesaw is an experiment in automating that research loop, with the goal of accelerating exploratory safety research on small models.

<br>

## License

[Apache 2.0](LICENSE)
