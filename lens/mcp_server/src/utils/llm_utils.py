from langchain_anthropic import ChatAnthropic

from ..config import ANTHROPIC_API_KEY, LENS_LLM_MAX_TOKENS, LENS_LLM_MODEL, LENS_LLM_TEMPERATURE


def make_llm() -> ChatAnthropic:
    """Build the ChatAnthropic client used by every workflow node.

    Centralised so all nodes (parse_plan, interpret_result, check_followup)
    share the same model config rather than re-instantiating it.
    """
    return ChatAnthropic(
        model=LENS_LLM_MODEL,
        api_key=ANTHROPIC_API_KEY,
        temperature=LENS_LLM_TEMPERATURE,
        max_tokens=LENS_LLM_MAX_TOKENS,
    )
