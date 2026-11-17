import logging

from app.core.llm import llm
from app.models.agent_models import RouterDecision

logger = logging.getLogger(__name__)

ROUTER_PROMPT = """
You are an intelligent query-routing classifier. Select exactly one agent:
- crypto_agent: cryptocurrency, Bitcoin, Ethereum, blockchain, token markets,
  crypto prices, or cryptocurrency trading.
- car_analyzer_agent: cars, vehicles, vehicle prices, specifications, fuel economy,
  vehicle comparisons, buying advice, or automotive performance.
- general_agent: all other topics, including programming, software engineering,
  education, science, and general technology.

Choose based on the main intent. If a question spans multiple topics, choose the
agent best suited to answer the user's primary request. Do not answer the question.
Return the original question, selected agent, and a concise routing reason.
"""


async def route_question(question: str) -> RouterDecision:
    """Classify a question using validated Pydantic structured output."""
    structured_llm = llm.with_structured_output(RouterDecision)
    decision = await structured_llm.ainvoke(
        [
            ("system", ROUTER_PROMPT),
            ("human", question),
        ]
    )
    if decision.user_question.strip() != question.strip():
        decision = decision.model_copy(update={"user_question": question})
    logger.info(
        "Question routed to %s: %s",
        decision.selected_agent.value,
        decision.routing_reason,
    )
    return decision
