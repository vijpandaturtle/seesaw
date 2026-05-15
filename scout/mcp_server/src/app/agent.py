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
from ..tools import SCOUT_TOOLS


def build_agent(checkpointer=None):
    """Build and return the Scout ReAct agent.

    Args:
        checkpointer: LangGraph checkpointer for memory. Defaults to in-process MemorySaver.

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

    agent = create_react_agent(
        model=model,
        tools=SCOUT_TOOLS,
        checkpointer=checkpointer,
        prompt=SystemMessage(content=SCOUT_SYSTEM_PROMPT),
    )

    return agent
