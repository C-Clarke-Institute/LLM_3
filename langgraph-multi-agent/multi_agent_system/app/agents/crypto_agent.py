from app.agents.tool_runner import run_tool_enabled_agent
from app.core.llm import llm
from app.tools.crypto_tools import CRYPTO_TOOLS

CRYPTO_SYSTEM_PROMPT = """
You are a careful cryptocurrency and blockchain specialist.
Answer the user's question clearly and distinguish general educational information
from current market information. When the user asks about current prices, market
capitalization, volume, or 24-hour performance, use the CoinGecko tools.
Never invent prices, market statistics, news, or predictions. State the source and
the fact that market data can change rapidly. Treat tool errors or missing data
as unavailable rather than filling gaps. Do not present investment advice as certainty.
"""


async def process_crypto_question(question: str) -> str:
    """Answer a cryptocurrency question with access to cryptocurrency tools."""
    return await run_tool_enabled_agent(
        llm=llm,
        system_prompt=CRYPTO_SYSTEM_PROMPT,
        question=question,
        tools=CRYPTO_TOOLS,
    )
