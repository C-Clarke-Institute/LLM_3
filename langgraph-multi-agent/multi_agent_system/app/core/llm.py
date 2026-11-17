from functools import lru_cache

from langchain_openai import ChatOpenAI

from app.core.config import settings


@lru_cache
def get_llm() -> ChatOpenAI:
    """Return the shared chat model instance used by all agents."""
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=settings.openai_temperature,
        timeout=settings.request_timeout_seconds,
        max_retries=2,
    )


llm = get_llm()
