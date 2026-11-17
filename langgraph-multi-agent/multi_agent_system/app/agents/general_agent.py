from app.agents.tool_runner import run_tool_enabled_agent
from app.core.llm import llm
from app.tools.general_tools import GENERAL_TOOLS

GENERAL_SYSTEM_PROMPT = """
You are a helpful, accurate general-purpose assistant. Answer questions outside
the cryptocurrency and automotive domains, including programming, software
engineering, education, science, and technology. Explain concepts clearly, adapt
the level of detail to the question, and state uncertainty when appropriate.
Use available tools only when they materially help answer the question.
"""


async def process_general_question(question: str) -> str:
    """Answer a general question using the shared LLM and optional general tools."""
    return await run_tool_enabled_agent(
        llm=llm,
        system_prompt=GENERAL_SYSTEM_PROMPT,
        question=question,
        tools=GENERAL_TOOLS,
    )
