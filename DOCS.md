# Seesaw — Setup, Usage, and Deployment

Seesaw is a three-agent pipeline for mechanistic interpretability research:

- **Scout** — takes a research question, searches arXiv and the web, produces a research plan
- **Lens** — runs TransformerLens experiments (logit lens, attention patterns, ablation, activation patching, direct logit attribution) specified by the plan
- **Quill** — critiques the results, identifies gaps, and generates follow-up experiment specs

An **orchestrator** chains the three together with two human-in-the-loop (HITL) checkpoints. Each agent is also independently runnable as an MCP server.

---

## Quick Start

```bash
# 1. Create and activate a virtualenv
uv venv .venv --python 3.12
source .venv/bin/activate

# 2. Install dependencies
uv pip install -e .

# 3. Add your API keys
cp .env.example .env
# edit .env — set ANTHROPIC_API_KEY (required) and FIRECRAWL_API_KEY (optional)

# 4. Run the full pipeline
python -m orchestrator.src.main --question "What attention heads mediate indirect object identification in GPT-2 Small?"
```

You'll be prompted to approve the Scout research plan, then to approve the Lens results before Quill critiques them. Output lands in `scout/outputs/`, `lens/outputs/`, and `quill/outputs/`.

---

## Environment Variables

Set in `.env` (see `.env.example`):

| Variable | Required | Used by |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | All three agents (Claude for reasoning/critique) |
| `FIRECRAWL_API_KEY` | No | Scout's `search_web` and `scrape_url` tools. Without it, those two tools no-op; `search_arxiv` still works. |

---

## Running the Orchestrator

```bash
python -m orchestrator.src.main --question "<research question>" [--skip-hitl]
```

| Flag | Effect |
|---|---|
| `--question`, `-q` | Required. The mech interp question for Scout to investigate. |
| `--skip-hitl` | Skip both approval checkpoints — runs Scout → Lens → Quill straight through with no prompts. Use for automated/batch runs. |

**Pipeline flow** (`orchestrator/src/pipeline.py`):

| Stage | Output | Checkpoint after |
|---|---|---|
| Scout | `research_plan.md` | Approve the plan? |
| Lens | `results_bundle.json` | Send results to Quill? |
| Quill | `critique_<timestamp>.json` / `.md` | — |

Each stage writes to disk before the next one reads from disk — there's no in-memory handoff. This means you can inspect intermediate output, or re-run a single stage without repeating the others (see below).

### Running a single stage

Each orchestrator client can be called directly if you don't need the full pipeline:

```python
from orchestrator.src.clients.scout_client import run_scout
from orchestrator.src.clients.lens_client import run_lens
from orchestrator.src.clients.quill_client import run_quill

plan_path = run_scout("What heads mediate IOI in GPT-2 Small?")
bundle, bundle_path = run_lens(plan_path.read_text())
report, report_path = run_quill(bundle_path)
```

`scout_client.py` and `lens_client.py` call the agent's LangGraph workflow **in-process** — they do not go through MCP. This is the fast path used by the orchestrator itself.

---

## Running Each Agent Standalone

Every agent has three ways to run it, from lightest to heaviest:

1. **Direct workflow call** — no MCP, no server process. Good for debugging the agent's core logic.
2. **MCP server (in-memory)** — a `fastmcp.Client` connects to the server object in the same Python process. No network involved. This is what `lens/mcp_client/` uses.
3. **MCP server (stdio/http)** — the server runs as its own process; any MCP-compatible client (Claude Desktop, another agent) can connect to it.

### Scout

```bash
# Direct
python3 -c "
from scout.mcp_server.src.app.agent import build_agent
agent = build_agent()
for chunk in agent.stream(
    {'messages': [{'role': 'user', 'content': 'YOUR QUESTION'}]},
    config={'configurable': {'thread_id': 'test-1'}},
    stream_mode='values',
):
    print(chunk['messages'][-1].content)
"

# As an MCP server (stdio)
python -m scout.mcp_server.src.server
```

Tools exposed: `search_arxiv`, `search_arxiv_web`, `search_web`, `scrape_url`, `save_research_plan`.
Resource: `system://status`. Prompt: `research_instructions`.

### Lens

```bash
# Direct (full research-plan workflow)
python3 -c "
from lens.mcp_server.src.workflows.lens_workflow import build_lens_graph
graph = build_lens_graph()
result = graph.invoke({'research_plan': open('scout/outputs/research_plan.md').read()})
print(result['bundle'])
"

# As an MCP server (stdio)
python -m lens.mcp_server.src.server

# Interactive MCP client (REPL, in-memory transport)
python -m lens.mcp_client.src.client
```

The REPL supports:
```
/tools                      list available MCP tools
/resources                  list available MCP resources
/prompts                    list available MCP prompts
/prompt/<name>               fetch a prompt and inject it into the conversation
/resource/<uri>               read and print an MCP resource
/model-thinking-switch      toggle printing of intermediate tool calls
/quit                       exit
<anything else>             sent to the agent, which may call MCP tools
```

Tools exposed: `logit_lens`, `attention_pattern`, `ablation`, `activation_patching`, `direct_logit_attribution` — each independently callable with a `model_name` and prompts, not just through the full workflow.
Resource: `results://last-bundle`. Prompt: `tool_selection_guide`.

### Quill

```bash
# Direct
python3 -c "
from quill.mcp_server.src.workflows.critique_workflow import build_quill_graph
from quill.mcp_server.src.models.schemas import ExperimentBundle
from quill.mcp_server.src.config import OUTPUTS_DIR

bundle = ExperimentBundle.from_json('lens/outputs/results_bundle.json')
graph = build_quill_graph()
result = graph.invoke({'bundle': bundle, 'output_dir': OUTPUTS_DIR})
print(result['critique_report'].overall_assessment)
"

# As an MCP server (stdio)
python -m quill.mcp_server.src.server
```

Tool exposed: `critique_experiment_bundle` (takes the results bundle as a JSON string, returns the full critique report — the four-node workflow isn't independently useful out of sequence, so it's a single tool rather than one-per-node).
Resource: `system://status`. Prompt: `critique_instructions`.

*(Quill doesn't yet have an interactive `mcp_client/` REPL — only Lens does. Copy `lens/mcp_client/` as a template if you need one.)*

---

## Directory Structure

All three agents share the same internal layout:

```
<agent>/mcp_server/src/
├── app/          # agent-specific core logic (ReAct agent, model loading, sandboxing, formatting)
├── config/       # settings.py (env vars, paths) + prompts.py (system prompts)
├── db/           # placeholder — no persistence layer yet
├── models/       # Pydantic/dataclass schemas
├── nodes/        # LangGraph node functions (Lens, Quill only — Scout is a ReAct agent, no nodes)
├── prompts/      # MCP prompt registrations
├── resources/    # MCP resource registrations
├── routers/      # wires tools/resources/prompts onto the FastMCP instance
├── tools/        # standalone callable tools (Scout, Lens) — Quill has none
├── ui/           # placeholder — no UI yet
├── utils/        # shared helpers
├── workflows/    # LangGraph graph builder (Lens, Quill)
└── server.py     # create_mcp_server() — entry point for MCP transport
```

`<agent>/mcp_client/` (currently only under `lens/`) is a separate package — the interactive REPL that connects to `server.py`'s FastMCP instance.

---

## Testing a Server Without an LLM

To confirm an MCP server's tools/resources/prompts are wired correctly without spending API credits, connect a bare `fastmcp.Client` and call things directly:

```python
import asyncio
from fastmcp import Client
from scout.mcp_server.src.server import mcp as scout_server  # or lens / quill

async def main():
    async with Client(scout_server) as client:
        print(await client.list_tools())
        print(await client.list_resources())
        result = await client.read_resource("system://status")
        print(result[0].text)

asyncio.run(main())
```

This exercises the actual MCP protocol layer (unlike calling the Python functions directly), which is where router wiring bugs show up.

---

## Deployment

Seesaw currently has no deployed/hosted component — everything runs locally via the CLI or as MCP servers over stdio. There is no HTTP server, no database, and no persisted state beyond the JSON/Markdown files each agent writes to its own `outputs/` directory.

### Running an MCP server as a persistent process

By default `mcp.run()` in each `server.py` uses stdio transport, meant for a parent process (like Claude Desktop or another agent) to spawn and talk to over stdin/stdout. To expose a server over the network instead, change the transport in `server.py`:

```python
if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8001)
```

Then run it as a long-lived process (e.g. under `systemd`, `supervisord`, or a container):

```bash
python -m lens.mcp_server.src.server
```

### Registering with an external MCP client

For stdio transport (e.g. Claude Desktop's `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "lens": {
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "lens.mcp_server.src.server"],
      "cwd": "/path/to/seesaw"
    }
  }
}
```

### Things to set up before any real deployment

None of the following exist yet — treat this as a checklist, not a description of current behavior:

- **Secrets management** — `.env` works locally; a deployed version needs the API keys injected via the hosting platform's secret store, not a checked-in file.
- **Persistence** — every agent's `db/` folder is a placeholder. Outputs are plain files on local disk (`*/outputs/`). Multi-instance or ephemeral-filesystem deployment (e.g. serverless) will lose this data unless you add real storage.
- **Concurrency** — `lens/mcp_server/src/app/model_session.py` caches loaded TransformerLens models in an in-process dict. Running multiple instances means each reloads models independently; there's no shared cache.
- **Timeouts / resource limits** — Lens's sandbox (`app/sandbox.py`) has a configurable timeout (`SANDBOX_TIMEOUT`, default 300s) per experiment, but nothing bounds total memory or GPU usage. TransformerLens experiments can be memory-heavy on larger models — size the host accordingly.
- **Monitoring** — no logging/tracing integration currently wired in (no equivalent of the `course/` reference implementations' Opik hooks).

---

## Troubleshooting

**`ModuleNotFoundError` for `lens`, `scout`, `quill`, `orchestrator`, or `shared`** — these are namespace packages (no top-level `__init__.py`), so they only resolve correctly when the project root is on `sys.path`. Run commands from the repo root, or with `python -m <package>.<module>` rather than `python <path>/file.py`.

**Scout's `search_web` / `search_arxiv_web` return `N/A` titles** — check `FIRECRAWL_API_KEY` is set. Firecrawl's `search()` returns a `SearchData` object with results under `.web`, not `.data` — if you're extending these tools, don't iterate over the response object directly (it's a pydantic model, so `for r in response` silently yields `(field_name, value)` pairs instead of raising an error).

**arXiv search hits `HTTP 429`** — the official arXiv API rate-limits aggressively on rapid repeated calls. Space out test runs or reduce `max_results`.

**Quill's saved report filename is unpredictable** — `save_critique.py` writes timestamped files (`critique_<YYYYMMDD_HHMMSS>.json/.md`), not a fixed name. Use the path returned by the workflow's `critique_path` state key rather than assuming a filename.
