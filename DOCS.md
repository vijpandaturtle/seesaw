# Seesaw — Setup, Usage, and Deployment

Seesaw is a three-agent pipeline for mechanistic interpretability research:

- **Scout** — takes a research question, searches arXiv and the web, produces a research plan
- **Lens** — runs TransformerLens experiments (logit lens, attention patterns, ablation, activation patching, direct logit attribution) specified by the plan
- **Quill** — critiques the results, identifies gaps, and generates follow-up experiment specs

An **orchestrator** chains the three together with two human-in-the-loop (HITL) checkpoints. Each agent is also independently runnable as an MCP server. An **eval suite** (`eval/`) grades all three agents — and grades its own graders first (see [Evaluation](#evaluation)).

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
| `DATABASE_URL` | No | Job store. Unset means a local SQLite file, which is all local development needs. |

Deployment adds several more — see [Deployment → Environment reference](#environment-reference).

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

## Evaluation

The eval suite lives in `eval/` and is built in three layers. The core rule: **fixtures test the graders; tasks test the agents** — and grader trust is established (via fixtures) before any agent score is taken seriously.

```
eval/
├── graders/      # 29 pure grading functions (scout.py, lens.py, quill.py)
│   ├── base.py               # GraderResult, judge() (LLM-as-judge, claude-opus-4-8)
│   └── langsmith_adapter.py  # wrap any grader into a LangSmith evaluator
├── fixtures/     # 56 canned artifacts with known verdicts — the grader meta-eval
├── tasks/        # 15 live-agent benchmark tasks (*.yaml) + loader
├── rubrics/      # 5 markdown rubrics used by llm_rubric graders
└── runner/       # executes tasks: agent adapters, grader dispatch, LangSmith upload
```

### Layer 1 — Graders (`eval/graders/`)

A grader is a pure function: `(artifact, reference) -> GraderResult` (normalized score in `[0,1]`, raw metrics, one-line detail). It never runs an agent. Three kinds:

| Kind | Mechanism | Examples |
|---|---|---|
| `code` | deterministic rule, no model call | `lens.numeric_correctness` (expected heads in top-k), `lens.robustness` (pass@k / pass^k across prompt variations), `quill.followup_executable` |
| `llm` | LLM-as-judge over free text | `scout.hypothesis_specificity`, `lens.causal_correlational_honesty`, `quill.gap_recall` |
| `human` | packages a review payload for an expert; no auto-score | `scout.researcher_would_run`, `quill.matches_expert` |

All 29 are enumerable via `eval.graders.REGISTRY`. The judge model is set in `eval/graders/base.py` (`JUDGE_MODEL`); note it takes no `temperature` — Opus 4.8 rejects sampling params.

### Layer 2 — Fixtures (`eval/fixtures/`): grading the graders

Each fixture is a canned artifact plus the verdict its grader **must** produce (`expected: pass | fail | na`). Every auto-gradable grader has at least one pass **and** one fail fixture, so a grader that always says yes (or always says no) is caught immediately. LLM-grader fixtures use a dead-band: pass requires score ≥ 0.6, fail requires ≤ 0.4; anything between is `indeterminate` and flags the fixture or judge for review rather than flapping.

```bash
# Code-grader fixtures — free, offline, run whenever you touch a grader
python -c "from eval.fixtures import check_offline; print(check_offline()['mismatches'] or 'all green')"

# LLM-grader fixtures — real judge calls (~24), run when you change the judge or rubrics
python -c "from dotenv import load_dotenv; load_dotenv(); from eval.fixtures import check_llm; print(check_llm(agent='scout'))"
```

A mismatch means the grader **or the fixture label** is wrong — read the judge's reasoning in the result before deciding which (this has happened: an ambiguous fixture scored exactly 0.5 and the dead-band caught it).

### Layer 3 — Tasks (`eval/tasks/`) and the runner (`eval/runner/`)

A task is a YAML spec: a unit of work for the live agents plus the graders to apply to whatever they produce. There is no per-task `expected:` — the score *is* the measurement. Balance comes from paired opposites instead: 8 ground-truth circuit tasks (IOI, greater-than, gender bias, induction, successor, copy suppression, ROME, docstring) and 7 adversarial probes (tool-bias questions Scout should refuse to force-fit, a causal-language trap for Lens, weak/strong calibration bundles for Quill).

Grader `type` vocabulary (validated by `eval/tasks/loader.py`):

| `type` | Checks | Config keys |
|---|---|---|
| `deterministic_tests` | named pytest files under `eval/tests/` | `required` |
| `llm_rubric` | judge + a rubric from `eval/rubrics/` | `rubric`, optional `target`, `model` |
| `static_analysis` | linters over produced code | `commands`, optional `paths` |
| `state_check` | artifacts/state (plan saved, bundle schema, assessment value) | `expect` |
| `tool_calls` | trajectory contains / avoids tool calls (glob params) | `required`, optional `forbidden` |

Handlers never silently pass: a missing test file or unimplemented `state_check` key surfaces as an explicit error result.

```bash
python -m eval.runner --list                          # all tasks
python -m eval.runner quill_weak_bundle_1             # run one locally → outputs/eval_runs/*.json
python -m eval.runner.langsmith_runner --all-cheap    # run non-pipeline tasks as a LangSmith experiment
```

The LangSmith runner registers every task as an example in the `seesaw-tasks` dataset, executes agents via `run_task`, and fans each grader result out as its own feedback score — so the experiment table shows one column per criterion, comparisons across experiments give per-criterion regressions, and `LANGSMITH_TRACING` nests the full agent trace under each row. Requires `LANGSMITH_API_KEY` in `.env`. Persisted runs are also readable straight from `outputs/eval_runs/*.json`.

### Adding to the suite

- **New grader** — write the pure function in `eval/graders/<agent>.py`, register it in `REGISTRY`, then add a pass and a fail fixture before trusting it.
- **New fixture** — append to `eval/fixtures/<agent>_fixtures.yaml` (`grader`, `expected`, `inputs`, tag `[llm]` if it needs the judge); rerun the checks above.
- **New task** — drop a YAML in `eval/tasks/` (the loader validates shape, grader types, and metric names on load); reference only rubrics that exist in `eval/rubrics/`.

### Known gaps

- The `deterministic_tests` files live in `eval/tests/`. They assert published findings (IOI name movers, ROME's mid-layer localisation, and so on) against the run's artifacts, and read those artifacts from environment variables the runner sets — so they can also be run directly with `SEESAW_EVAL_BUNDLE=<bundle.json> pytest eval/tests -q`.
- Token usage (`n_total_tokens`) is captured for Scout but not Lens/Quill (their graphs are `invoke()`d; per-call usage is visible in LangSmith traces instead).
- Human graders (`researcher_would_run`, `matches_expert`) export review payloads but no expert scores have been recorded yet.

---

## Deployment

Seesaw runs across three services. Nothing is always-on: the dashboard is
serverless and the agents are spawned per stage, so both scale to zero between
jobs.

| Service | Runs | Why there |
|---|---|---|
| **Vercel** | The dashboard (`seesaw-web`) | Reads and writes the job store; never executes an agent |
| **Modal** | Scout, Quill, Lens's graph, and the five experiment tools | A stage takes minutes to an hour, well past any serverless request limit |
| **Neon** | The job store | Vercel and Modal are different machines, so a local SQLite file can't be shared |

The dashboard inserts a queued row and calls Modal's `start` endpoint, which
`spawn()`s the stage and returns in milliseconds while the work continues. That
is what lets a serverless frontend drive hour-long jobs with no always-on worker
and no polling loop.

Artifacts go to a Modal volume rather than the container filesystem, which is
discarded when a function returns, and are served back over HTTP for the
dashboard to proxy.

### 1. Database

Create a Postgres database (Neon works well) and put its connection string in
`.env`:

```bash
DATABASE_URL=postgresql://…
```

Use the **pooled** endpoint — it suits short-lived serverless connections. The
schema is created on first connect; there is no migration step.

`DATABASE_URL` is what selects Postgres. Without it the store is a local SQLite
file, so local development needs no database at all. `SEESAW_DB_PATH` forces
SQLite even when `DATABASE_URL` is set, which is how CI avoids writing into
production.

### 2. Modal secret

Both Modal apps read one secret. `RUNNER_KEY` is generated here — it guards the
endpoint that starts jobs.

```bash
modal secret create seesaw-secrets \
  ANTHROPIC_API_KEY=… \
  DATABASE_URL=… \
  FIRECRAWL_API_KEY=… \
  RUNNER_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(18))')
```

### 3. Modal apps

```bash
modal deploy lens/modal_app.py          # GPU: the five experiment tools
modal deploy orchestrator/modal_app.py  # CPU: Scout, Quill, Lens's graph
```

The runner's first deploy takes a few minutes — it installs CPU torch. It
prints two URLs; keep both:

```
https://<account>--seesaw-runner-start.modal.run
https://<account>--seesaw-runner-artifact.modal.run
```

Two volumes are created automatically: `seesaw-hf-cache` for model weights, so
cold starts don't re-download from HuggingFace, and `seesaw-artifacts` for
plans, bundles, critiques, and plots.

### 4. Dashboard

From the `seesaw-web` checkout:

```bash
vercel link
vercel env add DATABASE_URL         production   # same string as above
vercel env add MODAL_RUN_STAGE_URL  production   # the start URL
vercel env add MODAL_ARTIFACT_URL   production   # the artifact URL
vercel env add MODAL_INVOKE_TOKEN   production   # RUNNER_KEY
vercel env add WRITE_KEY            production   # gates writes; generate one
vercel deploy --prod
```

`WRITE_KEY` makes the dashboard read-only for anyone without it. Reads stay
open; creating, approving, cancelling, and deleting need the key, since each
spends Anthropic credit and GPU time. Hold it by visiting `?key=…` once, or send
an `x-seesaw-key` header.

**Use the project alias, not the deployment URL.** Vercel forces its own SSO on
deployment-specific URLs (`<project>-<hash>-<team>.vercel.app`) whatever your
settings, so the middleware never runs there. `vercel alias ls` shows the stable
one.

### 5. Check it

```bash
curl -s https://<your-app>.vercel.app/api/jobs                       # 200, open

curl -s -o /dev/null -w '%{http_code}\n' -X POST \
     -H 'content-type: application/json' -d '{"question":"x"}' \
     https://<your-app>.vercel.app/api/jobs                          # 403, no key
```

Then create a job in the dashboard and watch it move `queued → running →
awaiting_approval`.

### Environment reference

| Variable | Where | Purpose |
|---|---|---|
| `DATABASE_URL` | Modal secret, Vercel, `.env` | Job store. Unset locally means SQLite |
| `SEESAW_DB_PATH` | local, CI | Force SQLite even when `DATABASE_URL` is set |
| `SEESAW_ARTIFACTS_DIR` | Modal image | Redirects all three agents' outputs to the volume |
| `SEESAW_LENS_BACKEND` | Modal runner | `modal` sends experiments to the GPU app; default `local` |
| `RUNNER_KEY` | Modal secret | Guards the endpoint that starts jobs |
| `WRITE_KEY` | Vercel | Gates dashboard writes |
| `MODAL_RUN_STAGE_URL`, `MODAL_ARTIFACT_URL`, `MODAL_INVOKE_TOKEN` | Vercel | Point the dashboard at the runner |
| `SEESAW_REPO_DIR` | local only | Lets the dashboard spawn a local worker instead of calling Modal |

### Things that cost time to discover

- **Neon's pooled endpoint is PgBouncer in transaction mode.** Session state
  doesn't reliably carry between statements, so `SET search_path` silently stops
  applying — `current_schema()` came back NULL on a connection whose
  `search_path` looked correct. Anything session-scoped (`SET`, advisory locks,
  temp tables) is unsafe there; the store schema-qualifies its table instead.
- **Connection latency dominates.** Connect-per-query took 4.6s per read. A
  module-level pool, schema DDL once per pool rather than per query, and
  autocommit — single-statement operations never needed a transaction — brought
  it to 276ms, about one round trip.
- **The CPU runner still needs torch.** `run_experiment` imports the tool
  registry, and reading a tool's signature imports modules that import torch and
  matplotlib. Install the CPU wheels; the forward passes still happen on the GPU
  app.
- **Agent output paths were derived from `__file__`.** On Modal that's a
  container discarded when the function returns, so artifacts vanished and the
  dashboard showed empty tabs. `SEESAW_ARTIFACTS_DIR` exists for this.
- **Device placement only fails on a GPU.** A tensor created without a device
  lands on CPU and breaks against model weights on `cuda`. Nothing local catches
  it.
- **arXiv rate-limits per client instance.** Constructing a fresh
  `arxiv.Client()` per call resets its politeness delay, which reliably produced
  HTTP 429 in deployment while surviving locally, where runs are spaced apart by
  hand.
- **`modal deploy` executes the app file locally**, so anything imported at
  module scope must be installed on your machine, not just in the image — keep
  `fastapi` and similar imports inside the functions.

### Running an MCP server as a persistent process

Separate from the above: each agent is also usable standalone over MCP. By
default `mcp.run()` in each `server.py` uses stdio transport, meant for a parent
process (Claude Desktop, or another agent) to spawn and talk to over
stdin/stdout. To expose one over the network instead:

```python
if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8001)
```

Then run it as a long-lived process under `systemd`, `supervisord`, or a
container:

```bash
python -m lens.mcp_server.src.server
```

---

## Troubleshooting

**`ModuleNotFoundError` for `lens`, `scout`, `quill`, `orchestrator`, or `shared`** — these are namespace packages (no top-level `__init__.py`), so they only resolve correctly when the project root is on `sys.path`. Run commands from the repo root, or with `python -m <package>.<module>` rather than `python <path>/file.py`.

**Scout's `search_web` / `search_arxiv_web` return `N/A` titles** — check `FIRECRAWL_API_KEY` is set. Firecrawl's `search()` returns a `SearchData` object with results under `.web`, not `.data` — if you're extending these tools, don't iterate over the response object directly (it's a pydantic model, so `for r in response` silently yields `(field_name, value)` pairs instead of raising an error).

**arXiv search hits `HTTP 429`** — the official arXiv API rate-limits aggressively on rapid repeated calls. Space out test runs or reduce `max_results`.

**Quill's saved report filename is unpredictable** — `save_critique.py` writes timestamped files (`critique_<YYYYMMDD_HHMMSS>.json/.md`), not a fixed name. Use the path returned by the workflow's `critique_path` state key rather than assuming a filename.
