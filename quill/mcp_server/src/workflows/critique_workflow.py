"""LangGraph workflow for the Quill critique agent.

Graph structure (linear — no loops):
    critique_experiments → identify_gaps → generate_followups → save_critique → END
"""

from pathlib import Path
from typing import Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from ..app.schemas import (
    CritiqueReport,
    ExperimentBundle,
    ExperimentCritiquesOutput,
    FollowUpsOutput,
    GapsOutput,
)
from ..nodes import (
    critique_experiments,
    generate_followups,
    identify_gaps,
    save_critique,
)


class QuillState(TypedDict):
    # Input
    bundle: ExperimentBundle
    output_dir: Path

    # Node outputs
    critiques_output: Optional[ExperimentCritiquesOutput]
    gaps_output: Optional[GapsOutput]
    followups_output: Optional[FollowUpsOutput]

    # Final
    critique_report: Optional[CritiqueReport]
    critique_path: Optional[Path]


def build_quill_graph(checkpointer=None):
    """Build and compile the Quill LangGraph critique workflow.

    Args:
        checkpointer: LangGraph checkpointer. Defaults to MemorySaver.

    Returns:
        Compiled StateGraph ready to invoke or stream.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    builder = StateGraph(QuillState)

    builder.add_node("critique_experiments", critique_experiments)
    builder.add_node("identify_gaps",        identify_gaps)
    builder.add_node("generate_followups",   generate_followups)
    builder.add_node("save_critique",        save_critique)

    builder.set_entry_point("critique_experiments")
    builder.add_edge("critique_experiments", "identify_gaps")
    builder.add_edge("identify_gaps",        "generate_followups")
    builder.add_edge("generate_followups",   "save_critique")
    builder.add_edge("save_critique",        END)

    return builder.compile(checkpointer=checkpointer)
