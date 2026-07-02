<div align="center">

# Seesaw

**A multi-agent framework for automated mechanistic interpretability research**

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![Built with LangGraph](https://img.shields.io/badge/built%20with-LangGraph-1c1c1c.svg)](https://github.com/langchain-ai/langgraph)
[![MCP](https://img.shields.io/badge/protocol-MCP-purple.svg)](https://modelcontextprotocol.io)

[Setup & Usage Docs](DOCS.md) · [Architecture](#architecture) · [Blog](https://mindsandai.substack.com)

</div>

<br>

<p align="center">
  <img src="assets/seesaw-v0.jpg" alt="Seesaw v0 Architecture" width="720">
</p>

<br>

## Overview

Seesaw automates the hypothesis → experiment → critique loop in mechanistic interpretability. A human researcher provides a research question; three agents handle the rest.

<div align="center">

**Scout** `research question → Research Plan` → **Lens** `Research Plan → ExperimentBundle` → **Quill** `ExperimentBundle → CritiqueReport`

</div>

Human-in-the-loop checkpoints sit between every stage, so you stay in control of what actually runs. Each agent is also independently usable as a standalone [MCP](https://modelcontextprotocol.io) server — see [DOCS.md](DOCS.md) for how to run each one on its own.

<br>

## Quick Start

```bash
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -e .

cp .env.example .env   # add ANTHROPIC_API_KEY (required), FIRECRAWL_API_KEY (optional)

python -m orchestrator.src.main --question "What attention heads mediate indirect object identification in GPT-2 Small?"
```

Full setup, per-agent usage, and deployment notes live in **[DOCS.md](DOCS.md)**.

<br>

## Agents

<table>
<tr>
<td width="33%" valign="top">

### 🔭 Scout
**The Research Planner**

Searches arXiv and the web, then produces a structured Research Plan — hypotheses, target models, and concrete experiments to run.

**Tools**
`search_arxiv` · `search_arxiv_web` · `search_web` · `scrape_url` · `save_research_plan`

**Stack**
LangGraph ReAct · Claude · Firecrawl

</td>
<td width="33%" valign="top">

### 🔬 Lens
**The Experiment Runner**

Executes mechanistic interpretability experiments from the Research Plan via TransformerLens, then generates an LLM interpretation of each result.

| Tool | Answers |
|---|---|
| `logit_lens` | Where does the model decide? |
| `attention_pattern` | What does each head attend to? |
| `direct_logit_attribution` | Which components write the prediction? |
| `ablation` | Is this component causally necessary? |
| `activation_patching` | Which layer/position carries the signal? |

**Stack**
LangGraph StateGraph · Claude · TransformerLens

</td>
<td width="33%" valign="top">

### 🪶 Quill
**The Reviewer**

Reviews the ExperimentBundle like a scientific peer reviewer — flags methodological issues, unsupported conclusions, and missing experiments. Outputs follow-up specs Lens can execute directly.

**Stack**
LangGraph StateGraph · Claude Opus

</td>
</tr>
</table>

<br>

## Architecture

Every agent shares the same internal structure — `app/`, `config/`, `models/`, `tools/`, `routers/`, `resources/`, `prompts/`, `server.py` — so the codebase reads consistently whether you're in Scout, Lens, or Quill. Details in [DOCS.md](DOCS.md#directory-structure).

The pipeline itself is deliberately simple: a fixed-order **prompt-chaining workflow**, not autonomous multi-agent routing. Scout always runs first, Lens second, Quill third — no agent decides who runs next. That keeps the system predictable and easy to debug, with room to grow into an Evaluator-Optimizer loop between Lens and Quill later.

<br>

## Motivation

Most mechanistic interpretability work is done by hand — identify a circuit, ablate it, write it up. Seesaw is an experiment in automating that research loop, with the goal of accelerating exploratory safety research on small models.

<br>

## License

[Apache 2.0](LICENSE)
