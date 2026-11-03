from langchain_anthropic import ChatAnthropic

from ..config import ANTHROPIC_API_KEY, LENS_LLM_MAX_TOKENS, LENS_LLM_MODEL


def make_llm(max_tokens: int = LENS_LLM_MAX_TOKENS) -> ChatAnthropic:
    """Build the ChatAnthropic client used by every workflow node.

    Centralised so all nodes (parse_plan, interpret_result, check_followup)
    share the same model config rather than re-instantiating it. No
    temperature: API version 2026-10-21 rejects temperature, top_p and top_k
    on every model, so every agent uses the default sampling.
    """
    return ChatAnthropic(
        model=LENS_LLM_MODEL,
        api_key=ANTHROPIC_API_KEY,
        max_tokens=max_tokens,
    )
