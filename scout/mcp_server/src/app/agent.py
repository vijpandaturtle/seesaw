from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from ..config import (
    ANTHROPIC_API_KEY,
    SCOUT_MODEL,
    SCOUT_MAX_TOKENS,
    SCOUT_TEMPERATURE,
    SCOUT_SYSTEM_PROMPT,
)
from ..tools import SCOUT_TOOLS, make_save_research_plan


def build_agent(checkpointer=None, output_dir=None):
    """Build and return the Scout ReAct agent.

    Args:
        checkpointer: LangGraph checkpointer for memory. Defaults to in-process MemorySaver.
        output_dir: Where save_research_plan writes. Defaults to Scout's OUTPUTS_DIR.

    Returns:
        Compiled LangGraph ReAct agent ready to stream or invoke.
    """
    model = ChatAnthropic(
        model=SCOUT_MODEL,
        api_key=ANTHROPIC_API_KEY,
        temperature=SCOUT_TEMPERATURE,
        max_tokens=SCOUT_MAX_TOKENS,
    )

    if checkpointer is None:
        checkpointer = MemorySaver()

    tools = SCOUT_TOOLS
    if output_dir is not None:
        tools = [t for t in SCOUT_TOOLS if t.name != "save_research_plan"]
        tools.append(make_save_research_plan(output_dir))

    agent = create_react_agent(
        model=model,
        tools=tools,
        checkpointer=checkpointer,
        prompt=SystemMessage(content=SCOUT_SYSTEM_PROMPT),
    )

    return agent
