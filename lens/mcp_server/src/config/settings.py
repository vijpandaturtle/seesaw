import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")

# ── Paths ─────────────────────────────────────────────────────────────────────
LENS_ROOT: Path = Path(__file__).resolve().parents[4]   # lens/
# SEESAW_ARTIFACTS_DIR redirects output to shared storage. Running on
# Modal, the container filesystem is discarded when the function ends,
# so artifacts have to land on a mounted volume to outlive the run.
_ARTIFACTS_BASE = os.getenv("SEESAW_ARTIFACTS_DIR")
OUTPUTS_DIR: Path = (
    Path(_ARTIFACTS_BASE) / "lens" if _ARTIFACTS_BASE else LENS_ROOT / "outputs"
)
PLOTS_DIR: Path = OUTPUTS_DIR / "plots"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Model ─────────────────────────────────────────────────────────────────────
LENS_LLM_MODEL: str = "claude-sonnet-4-5"
LENS_LLM_MAX_TOKENS: int = 2_000
LENS_LLM_TEMPERATURE: float = 0.0

# ── TransformerLens model loading ─────────────────────────────────────────────
TL_CENTER_WRITING_WEIGHTS: bool = True
TL_CENTER_UNEMBED: bool = True
TL_FOLD_LN: bool = True
TL_REFACTOR_FACTORED_ATTN: bool = True
# Note: do NOT set use_attn_result=True — incompatible with refactor_factored_attn_matrices

# ── Sandbox ───────────────────────────────────────────────────────────────────
SANDBOX_TIMEOUT: int = int(os.getenv("SEESAW_SANDBOX_TIMEOUT", "300"))   # seconds — ablation/patching can be slow on CPU
# Tools that re-run the model once per layer for every prompt. On a VLM with
# ~750-token image inputs that's hundreds of forward passes, far past 300s on
# a laptop.
SWEEP_TIMEOUT: int = int(os.getenv("SEESAW_SWEEP_TIMEOUT", "1800"))
SWEEP_TOOLS: dict[str, frozenset[str]] = {
    "seesaw": frozenset(),
    "lever":  frozenset({"image_token_ablation", "activation_patching"}),
}

# ── Workflow ──────────────────────────────────────────────────────────────────
MAX_FOLLOWUPS: int = 2       # max auto-generated follow-up experiments per run

# ── Neuronpedia ───────────────────────────────────────────────────────────────
# circuit_tracing generates attribution graphs on Neuronpedia, which stores the
# graph and its prompt under a public URL. Off unless SEESAW_NEURONPEDIA_PUBLISH=1:
# without it, circuit_tracing only reads graphs that already exist.
NEURONPEDIA_PUBLISH: bool = os.getenv("SEESAW_NEURONPEDIA_PUBLISH", "") == "1"
