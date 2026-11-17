from pydantic import BaseModel, Field

from app.models.agent_models import AgentName


class AgentResponse(BaseModel):
    """Stable response contract returned by the API."""

    success: bool
    question: str
    selected_agent: AgentName
    answer: str
    routing_reason: str
    error: str | None = Field(default=None, description="Safe, user-facing error.")
