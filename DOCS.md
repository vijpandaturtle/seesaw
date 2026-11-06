# Seesaw — Setup, Usage, and Deployment

Seesaw is a three-agent pipeline for mechanistic interpretability research:

- **Scout** turns a research question into a research plan. It chooses one of Seesaw's [pathways](#pathways) and that pathway's inputs, or plans free-form experiments when none fits. Literature search over arXiv and the web is opt-in.
- **Lens** runs the plan: the pathway first, then any extra experiments, using the [Lens tools](#lens-tools). Then it interprets each result with an LLM.
- **Quill** critiques the results, treating the pathway's verdict as the primary evidence. It identifies gaps and writes follow-up experiment specs Lens can run.

The agents currently run on Claude.

A **job** chains the three together with two human approval gates. Jobs are recorded in a job store that the `seesaw` CLI and the dashboard share. Each agent can also run on its own as an MCP server. An **eval suite** (`eval/`) grades all three agents, and it checks its own graders first (see [Evaluation](#evaluation)).

---

## Quick Start

```bash
# 1. Create and activate a virtualenv
uv venv .venv --python 3.12
source .venv/bin/activate

# 2. Install, with the experiment stack Lens needs (torch, TransformerLens, SAELens)
uv pip install -e ".[experiments]"

# 3. Add your API key
cp .env.example .env
# edit .env: set ANTHROPIC_API_KEY (required), and HF_TOKEN to run the gated Gemma or Llama models

# 4. Run a job on a question with a known answer (Qwen2.5 is open: no HF token needed)
seesaw jobs submit "Does the model use the gender a job title implies to choose 'she' or 'he'?" \
  --pathway concept_direction --model Qwen/Qwen2.5-1.5B --follow
```

With `--follow`, the job's log streams in your terminal and you're asked to approve at each gate: first Scout's plan, pathway and inputs, then the Lens results before Quill critiques them. The job runs in a detached local worker, so Ctrl-C stops watching but not the job. Artifacts go to `outputs/jobs/<id>/`.

A plain `uv pip install -e .` installs only Scout, Quill and the orchestrator, which don't need torch. It's enough for `seesaw plan`, `seesaw critique` and the offline eval checks, but not for anything that runs Lens.

| Extra | Adds | Needed for |
|---|---|---|
| `experiments` | torch, TransformerLens, SAELens, matplotlib | Lens, the pathways, `seesaw.interp` |
| `lever` | [lever-interp](https://github.com/vijpandaturtle/lever-interp), pillow | Vision-language models (`SEESAW_INTERP_BACKEND=lever`); install with `experiments` |
| `eval` | LangSmith, pytest | The eval suite and the test suite |
| `modal` | Modal | Running Lens experiments on Modal |
| `postgres` | psycopg | A hosted job store (`DATABASE_URL`) |
| `all` | All of the above except `lever` | |

---

## Environment Variables

Set in `.env` (see `.env.example`):

| Variable | Required | Used by |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | All three agents, and the eval suite's LLM judge |
| `FIRECRAWL_API_KEY` | No | Scout's `search_web` and `scrape_url`, used only with `--search`. Without it those two return nothing, and `search_arxiv` still works. |
| `LANGSMITH_API_KEY` | No | The LangSmith eval runner and tracing |
| `NEURONPEDIA_API_KEY` | No | Neuronpedia's write endpoints. Feature labels and reads need no key. |
| `SEESAW_NEURONPEDIA_PUBLISH` | No | `1` lets `circuit_tracing` generate attribution graphs on Neuronpedia. Generated graphs and their prompts are public. |
| `SEESAW_INTERP_BACKEND` | No | `lever` runs Lens on vision-language models; default `seesaw` (see [Interp Backends](#interp-backends-seesaw-and-lever)) |
| `DATABASE_URL` | No | Job store. Unset means a local SQLite file at `outputs/jobs.db`, which is all local use needs. |

Deployment adds several more; see [Deployment → Environment reference](#environment-reference).

---

## The `seesaw` CLI

`seesaw` is installed with the package (`seesaw/cli.py`). Every subcommand is a thin call into the library, so anything it does, a script or notebook can do too.

| Command | Does |
|---|---|
| `seesaw jobs …` | Submit and drive jobs in the job store, the main way to run Seesaw ([below](#jobs)) |
| `seesaw run "<question>" [--skip-hitl]` | The full Scout → Lens → Quill pipeline in this process. Records no job, uses no pathway, and Scout searches the literature. |
| `seesaw plan "<question>"` | Scout only: question → research plan |
| `seesaw critique <results_bundle.json>` | Quill only: results bundle → critique report |
| `seesaw eval <task_id>` · `seesaw eval --list` | Run one eval task against the live agents, or list the tasks |
| `seesaw fixtures [--llm [AGENT]]` | The grader meta-eval. It's offline and free by default, and `--llm` runs the judge fixtures, which cost API calls. |

`python -m orchestrator.src.main --question "<question>" [--skip-hitl]` is the same as `seesaw run`.

### Jobs

```bash
seesaw jobs submit "<question>" [--pathway auto|none|<name>] [--model M ...] [--other-model M] [--search] [--auto-approve] [--follow]
```

A job runs Scout, stops at a gate for your approval, runs Lens, stops at a second gate, then runs Quill. The job store records the job, so with a Postgres store (`DATABASE_URL`) a job submitted here shows in the dashboard, and one queued from the dashboard can be approved here. The dashboard reads only Postgres: a job in the local SQLite file (no `DATABASE_URL`, or `SEESAW_DB_PATH` set) shows only in the CLI. For trial runs, set `SEESAW_DB_SCHEMA` (on both sides) to use an isolated copy of the tables.

| Command | Does |
|---|---|
| `submit "<question>"` | Create a job and start its Scout stage. Options below. |
| `list` | Recent jobs, newest first. `--limit`, `--json` |
| `show <id>` | Status, artifact paths, error, and what to do next. `--json` |
| `approve <id>` | Clear the approval gate and start the next stage. Refused unless the job is awaiting approval. |
| `retry <id>` | Re-run a failed stage. Refused unless the job has failed. A retried Scout stage chooses its pathway again. |
| `rerun <id>` | Run a job's pathway again on its approved inputs, as a new job starting at Lens (Scout doesn't run again). `--model` runs it on another model. See [docs/use-cases.md](docs/use-cases.md#reproducing-a-result). |
| `cancel <id>` | Stop the job |
| `logs <id>` | Worker output. `-n` lines, `-f` streams until the job stops. |
| `open <id>` | Open the job in the dashboard at `SEESAW_WEB_URL` (default `http://localhost:3000`) |

`submit` options:

| Option | Effect |
|---|---|
| `--pathway auto` (default) | Scout reads every pathway's summary and picks the one that answers the question, with a one-sentence reason. If none fits, it picks none and plans the experiments itself. |
| `--pathway <name>` | That pathway runs, and Scout only chooses its inputs. Use this when you know which one you want, and always for evaluation. |
| `--pathway none` | No pathway; Scout plans freely |
| `--model` | A model Lens runs on. Repeat it to run the pathway on several models, in order. The default is `google/gemma-2-2b`. The supported models are listed in [`shared/lens_tools.py`](shared/lens_tools.py). |
| `--other-model` | `model_diffing` only: the changed model to compare the first one with |
| `--search` | Let Scout search arXiv and the web while planning. It's off by default, and without it Scout cites no papers as sources. |
| `--auto-approve` | Run all stages without stopping at the gates |
| `--follow`, `-f` | Stream the log, and ask whether to approve at each gate |
| `--runner` | Who runs the stage: `auto` (default), `local`, `modal` or `queue` |

`submit`, `approve`, `retry` and `rerun` all accept `--runner` and `--follow`.

Scout's pathway inputs are checked before the gate. The lists must line up, and every answer must be a single token for the model. If they don't, Scout gets the problems back to fix, up to twice. At the gate you see the plan, the pathway Scout chose and why, and its inputs.

**Who runs a stage.** By default (`--runner auto`), the stage runs on the Modal function when `MODAL_RUN_STAGE_URL` is set. Otherwise it runs in a detached local worker that writes to `outputs/job_logs/<id>.log`. `--runner queue` starts nothing and leaves the job for `python -m orchestrator.src.worker --poll`. The commands use whatever store `DATABASE_URL` points at, so if `.env` holds the production URL, `submit` queues a real job.

**Where artifacts go.** Each job has its own folder, `outputs/jobs/<id>/`, holding the plan, the results bundle, the pathway report, plots and the critique. Under `SEESAW_ARTIFACTS_DIR` it's `$SEESAW_ARTIFACTS_DIR/jobs/<id>/`. `seesaw run`, `plan` and `critique` write to each agent's own `scout/outputs/`, `lens/outputs/` and `quill/outputs/` instead.

#### The job timeline

Besides the log, every worker records a structured timeline of the job in the store's `job_events` table ([shared/db/events.py](shared/db/events.py)), written as it happens:

| Kind | One per |
|---|---|
| `stage` | stage (`scout`, `lens`, `quill`, `pathway`), with how long it took and the error if it failed |
| `node` | LangGraph node the agent ran |
| `llm` | model call: the model, tokens in and out, the last prompt message, the answer and any tool calls it asked for |
| `tool` | agent tool call (Scout's searches, its plan save): input and output |
| `experiment` | Lens experiment: tool, prompts, arguments, status, plots, raw data |
| `pathway` / `step` | pathway run and each of its steps, with the step's numbers and plots |
| `check` | check on Scout's pathway inputs, with the problems it sent back |
| `error` / `status` | failure with its traceback, and every change of job status |
| `cost` | finished stage: what it cost in Claude tokens, GPU time and runner time ([shared/costs.py](shared/costs.py)) |

Events nest under the span they happened in, so an LLM call sits under its node and the node under its stage. The dashboard shows this as the job's **Timeline** tab. Long strings and lists are cut short; the artifacts on disk keep everything. Nothing is recorded outside a worker, and a timeline that can't be written prints one warning and never fails the job. The table is created the first time a current worker opens the store, like the other columns.

#### Cost at each gate

When a job stops at an approval gate, the worker writes its `cost` column: what the job has spent so far, stage by stage, and an estimate for the stage you are about to approve. The dashboard shows both in the approval box, and `seesaw jobs show` and the `--follow` prompt print them. Spent amounts come from the job's own timeline: every LLM call's tokens at Anthropic's list prices, every Lens experiment's GPU time when it ran on Modal (plus the minute it takes to load and the ten minutes the GPU container stays warm after, which Modal bills), and the runner's time. The estimate is the median of the last 20 finished stages of the same kind, scaled for Lens by how many Lens tools Scout's plan names. Before any stage has a `cost` record, it uses default figures measured on the first jobs, and says so. These are estimates at list prices. The Anthropic and Modal bills are the record. Prices live in `shared/costs.py` and need updating when either provider changes them.

```python
from shared.db import events
events.list_events("b1967577")     # oldest first, data decoded
```

---

## Pathways

A pathway is a fixed, hand-coded investigation that ends in a verdict ([`seesaw/pathways/`](seesaw/pathways/)). It's the expert baseline Lens is judged against. It also gives a job a result that doesn't depend on an LLM reading the plan well.

| Pathway | Question it answers | Inputs |
|---|---|---|
| `circuit_discovery` | What mechanism does the model use for this task? Nominate heads, verify heads and MLPs by patching, check the circuit is faithful. | `prompts`, `positive_tokens`, `negative_tokens`; optional `corrupted_prompts`, `reference_prompts` |
| `concept_direction` | Is a concept a direction the model uses, and can it be steered safely? Fit on half the pairs, remove and steer on the other half, check it bakes into the weights. | `prompts`, `counterfactual_prompts`, `positive_tokens`, `negative_tokens`; optional `control_prompts`, `control_texts` |
| `probe_monitor` | Would a probe on the activations make a good detector? Train, test on held-out texts, compare with text-only baselines. | `prompts`, `contrast_prompts`, `test_prompts`, `test_contrast_prompts` |
| `spurious_audit` | Does the answer depend on an attribute it shouldn't, and can LEACE erase it without breaking the task? | `prompts`, `counterfactual_prompts`, `positive_tokens`, `negative_tokens`; optional `control_texts` |
| `factual_recall` | Where does the model retrieve these facts? Causal tracing over layers and positions. | `prompts`, `subjects`, `positive_tokens` |
| `editing_comparison` | Is the layer where a fact is traced the best layer to edit it? ROME at every layer vs tracing. | `prompts`, `subjects`, `positive_tokens`, `target_tokens`, `paraphrase_prompts`, `neighborhood_prompts`, `neighborhood_tokens` |
| `model_diffing` | What did a fine-tune, quantisation or edit change, and in which weights? | `prompts`, `other_model` |
| `feature_explain` | Which SAE features drive this output, and what are they? Sweeps published residual SAEs (Gemma-2-2B or Qwen2.5-7B-Instruct only). | `prompts`, `positive_tokens`, `negative_tokens` |

`attribute_audit` is `concept_direction` under its older name. It's kept so earlier jobs can be rerun.

- **Answer tokens are single tokens.** Most pathways score one answer token against one distractor per prompt, so a good question has an answer the model gives as one next token, such as `" Mary"` vs `" John"`. `probe_monitor` and `model_diffing` take texts instead.
- **Verdicts are deterministic.** Each pathway has a fixed verdict vocabulary, and its random baselines are seeded. The same pathway version on the same model and inputs gives the same verdict. Each pathway carries a `version`, and results are only comparable within one.
- **Every step is labelled** with its evidence strength (correlational, estimate or causal) and whether its controls ran. See [`shared/evidence.py`](shared/evidence.py).
- **Known answers are tested.** [`tests/test_pathways_models.py`](tests/test_pathways_models.py) runs each pathway on a case whose answer is known from the literature, on Gemma-2-2B, Llama-3.2-1B and Qwen2.5-1.5B. It checks what holds across model families, not one model's head numbers. It's opt-in (`SEESAW_MODEL_TESTS=1`) and needs a GPU.

Sample questions for each pathway, sorted by what you want to find out, are in **[docs/use-cases.md](docs/use-cases.md)**. The input definitions Scout is given are in [`seesaw/pathways/catalog.py`](seesaw/pathways/catalog.py).

---

## Lens Tools

Lens's tools are listed once, in [`shared/lens_tools.py`](shared/lens_tools.py). Scout's plan template, Lens's prompts, Quill's follow-up prompt and the eval graders all read that list. Each tool is labelled by the kind of evidence it gives:

- **descriptive** shows what the model represents or attends to. It makes no causal claim.
- **estimate** is a gradient-based approximation of a causal effect. It ranks candidates cheaply, but its numbers should never be reported as effects.
- **causal** is an intervention: it changes the model and measures what follows.

**Text models** (`seesaw` backend, the default), 18 tools:

| Kind | Tools |
|---|---|
| descriptive | `logit_lens` · `attention_pattern` · `direct_logit_attribution` · `linear_probe` · `feature_search` · `counterfactual` |
| estimate | `attribution_patching` · `feature_attribution` · `edge_attribution_patching` · `circuit_tracing` (gemma-2-2b only, from Neuronpedia) |
| causal | `ablation` · `activation_patching` · `head_patching` · `circuit_evaluation` · `path_patching` · `steering_vector` · `directional_ablation` · `feature_clamping` |

Lens chains them nominate-then-verify. An estimate or descriptive tool picks candidates, and a causal tool tests them:

- `attribution_patching` or `direct_logit_attribution` → `head_patching` or `ablation` with `heads` → `circuit_evaluation`
- `edge_attribution_patching` → `path_patching`
- `counterfactual` → `linear_probe` → `directional_ablation` or `steering_vector`
- `feature_attribution` → `feature_clamping`

**Vision-language models** (`lever` backend), 6 tools: `logit_lens`, `clip_direct_effects` (descriptive), and `vision_ablation`, `image_token_ablation`, `activation_patching`, `clip_head_ablation` (causal). See [Interp Backends](#interp-backends-seesaw-and-lever).

The pathways use further library functions that aren't Lens tools: causal tracing, MLP patching, ROME, LEACE, probe transfer and weight-level model diffing. They're in `seesaw.interp`.

---

## Library API

Everything the CLI does is available from Python. `import seesaw` is fast. It loads torch and LangGraph only when you first use something that needs them.

```python
import seesaw

# Agents
seesaw.run_pipeline("What heads mediate IOI in GPT-2 Small?", skip_hitl=True)
plan_path           = seesaw.run_scout("…question…")
bundle, bundle_path = seesaw.run_lens(plan_path.read_text())
report, report_path = seesaw.run_quill(bundle_path)

# A pathway on fixed inputs, no agents and no job: what the known-answer tests do
from pathlib import Path
from seesaw.experiments import load_model
from seesaw.pathways import circuit_discovery

model = load_model("gpt2")
result = circuit_discovery.run(model, prompts, positive_tokens, negative_tokens, Path("out"))
print(result.verdict)
print(result.to_markdown())     # one page: verdict, the evidence behind it, then the details

# Single experiments by model name (the model is loaded on first use)
from seesaw import experiments
experiments.ablation("gpt2", prompts=[...], positive_tokens=[" Mary"], negative_tokens=[" John"])

# The interp toolkit itself, grouped by technique
from seesaw.interp.causal_interventions import ablation, causal_tracing
from seesaw.interp.model_editing import rome, leace

# Eval suite
from seesaw import evals
evals.check_offline()
evals.run_task(evals.load_tasks()["quill_weak_bundle_1"])
```

`seesaw.interp` is grouped by technique: `black_box`, `causal_interventions`, `circuit_finding`, `feature_extraction`, `model_diffing`, `model_editing`, `probing`, `steering`, `visualization`. Lens's MCP tools and the pathways both wrap it. `seesaw.vlm` is the vision-language counterpart, on lever.

### Running a single stage

The agent functions above are the orchestrator's own clients (`orchestrator/src/clients/`). `run_scout` and `run_lens` call the agent's LangGraph workflow **in-process**, without going through MCP. Each stage writes to disk before the next one reads from disk, with no in-memory handoff. That means you can inspect intermediate output, or re-run one stage without repeating the others.

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

`build_agent(search=False)` gives Scout only `save_research_plan`. That's what a job uses unless it was submitted with `--search`. Without search, Scout states its expectations as hypotheses and cites nothing as a source.

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

Tools exposed: `logit_lens`, `attention_pattern`, `ablation`, `activation_patching`, `direct_logit_attribution` each have their own MCP tool; `run_tool` runs any other tool in the catalog ([`shared/lens_tools.py`](shared/lens_tools.py)) by name, with its arguments as `tool_kwargs`. All are independently callable with a `model_name` and prompts, not just through the full workflow.
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

## Interp Backends: seesaw and lever

Lens runs experiments through one of two libraries, chosen per run with
`SEESAW_INTERP_BACKEND`:

| Backend | Models | Library | Tools |
|---|---|---|---|
| `seesaw` (default) | TransformerLens text models (GPT-2, Pythia, Gemma for circuit tracing) | `seesaw.interp` | 18: patching, circuits, probes, steering, SAE features |
| `lever` | Vision-language models: LLaVA, Qwen2.5-VL / Qwen3-VL, CLIP | `seesaw.vlm`, on [lever](https://github.com/vijpandaturtle/lever-interp) | 6: logit lens, vision ablation, image-token ablation, activation patching, CLIP direct effects, CLIP head ablation |

```bash
uv pip install -e ".[experiments,lever]"
SEESAW_INTERP_BACKEND=lever seesaw run "Which LM layers read the written word in a typographic attack on LLaVA?"
```

The switch changes everything that lists tools: Scout's plan template, Lens's
planner and follow-ups, Quill's follow-up prompt and the MCP `run_tool`, all
read the active catalog in `shared/lens_tools.py`. It's independent of
`SEESAW_LENS_BACKEND`, which controls *where* experiments run; the lever
backend runs locally only for now.

**Images.** lever experiments take `images` (and `corrupted_images` for
activation patching), one per prompt. A plan can give a file path (relative
paths resolve against `SEESAW_IMAGE_DIR`), an http(s) URL, or a synthetic
image: `text:<words>`, `solid:<colour>`, or any of those plus `+text:<words>`
to stamp words on it, which is a typographic attack. See `seesaw/vlm/inputs.py`.

**Answer tokens** for chat VLMs have no leading space (`"Yes"`, not `" Yes"`).
For the CLIP tools, `prompts` are the candidate labels.

**Not covered:** the eval suite's tasks and graders are GPT-2 circuit tasks
and stay on the seesaw catalog, so evals don't score lever runs yet.

---

## Directory Structure

```
seesaw/          # the library and CLI: cli.py, jobs_cli.py, experiments.py, evals.py
├── interp/      # the interp toolkit, grouped by technique (no agents)
├── pathways/    # the hand-coded pathways, and catalog.py (their inputs)
└── vlm/         # vision-language tools on lever
scout/ lens/ quill/   # the three agents, each an MCP server (layout below)
orchestrator/    # pipeline.py (seesaw run), worker.py (runs job stages), clients/, modal_app.py
shared/          # lens_tools.py (the tool catalog), evidence.py, db/ (job store and timeline)
eval/            # graders, fixtures, tasks, rubrics, runner, and eval/tests/
tests/           # the unit and known-answer test suite (pytest tests/)
docs/            # use-cases.md, research-methodology.md
```

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
│   ├── base.py               # GraderResult, judge() (LLM-as-judge)
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

All 29 are enumerable via `eval.graders.REGISTRY`. The judge model is set in `eval/graders/base.py` (`JUDGE_MODEL`); note it takes no `temperature`, because the judge model rejects sampling params.

### Layer 2 — Fixtures (`eval/fixtures/`): grading the graders

Each fixture is a canned artifact plus the verdict its grader **must** produce (`expected: pass | fail | na`). Every auto-gradable grader has at least one pass **and** one fail fixture, so a grader that always says yes (or always says no) is caught immediately. LLM-grader fixtures use a dead-band: pass requires score ≥ 0.6, fail requires ≤ 0.4; anything between is `indeterminate` and flags the fixture or judge for review rather than flapping.

```bash
# Code-grader fixtures: free, offline, run whenever you touch a grader
seesaw fixtures

# LLM-grader fixtures: real judge calls (~24), run when you change the judge or rubrics
seesaw fixtures --llm          # all agents
seesaw fixtures --llm scout    # one agent
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
seesaw eval --list                                    # all tasks
seesaw eval quill_weak_bundle_1                       # run one locally → outputs/eval_runs/*.json
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
| **Modal** | Scout, Quill, Lens's graph, and the experiment tools | A stage takes minutes to an hour, well past any serverless request limit |
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
modal deploy lens/modal_app.py          # GPU: the experiment tools
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
| `SEESAW_INTERP_BACKEND` | local | `lever` runs Lens on vision-language models; default `seesaw` (see [Interp Backends](#interp-backends-seesaw-and-lever)) |
| `SEESAW_IMAGE_DIR` | local | Where relative image paths in a plan resolve (lever backend) |
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

**`ModuleNotFoundError: No module named 'transformer_lens'` (or `torch`)** means the experiment stack isn't installed. Lens, the pathways and `seesaw run` need it: `uv pip install -e ".[experiments]"`.

**`ModuleNotFoundError` for `lens`, `scout`, `quill`, `orchestrator`, or `shared`** — these are namespace packages (no top-level `__init__.py`), so they only resolve correctly when the project root is on `sys.path`. Run commands from the repo root, or with `python -m <package>.<module>` rather than `python <path>/file.py`.

**Scout's `search_web` / `search_arxiv_web` return `N/A` titles** — check `FIRECRAWL_API_KEY` is set. Firecrawl's `search()` returns a `SearchData` object with results under `.web`, not `.data` — if you're extending these tools, don't iterate over the response object directly (it's a pydantic model, so `for r in response` silently yields `(field_name, value)` pairs instead of raising an error).

**arXiv search hits `HTTP 429`** — the official arXiv API rate-limits aggressively on rapid repeated calls. Space out test runs or reduce `max_results`.

**Quill's saved report filename is unpredictable** — `save_critique.py` writes timestamped files (`critique_<YYYYMMDD_HHMMSS>.json/.md`), not a fixed name. Use the path returned by the workflow's `critique_path` state key rather than assuming a filename.
