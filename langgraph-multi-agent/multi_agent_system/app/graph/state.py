from typing import TypedDict

from app.models.agent_models import AgentName


class AgentState(TypedDict, total=False):
    user_question: str
    selected_agent: AgentName
    routing_reason: str
    agent_response: str
    error: str | None
