import logging

from app.graph.workflow import workflow
from app.models.agent_models import AgentName
from app.models.response_models import AgentResponse

logger = logging.getLogger(__name__)


class AgentService:
    """Application service that invokes the LangGraph workflow."""

    async def ask(self, question: str) -> AgentResponse:
        initial_state = {
            "user_question": question,
            "selected_agent": AgentName.GENERAL,
            "routing_reason": "",
            "agent_response": "",
            "error": None,
        }
        try:
            result = await workflow.ainvoke(initial_state)
            answer = result.get("agent_response")
            selected = result.get("selected_agent")
            reason = result.get("routing_reason", "")
            if not answer or not selected:
                raise RuntimeError("Workflow completed without a valid response.")
            if isinstance(selected, str):
                selected = AgentName(selected)
            return AgentResponse(
                success=True,
                question=question,
                selected_agent=selected,
                answer=answer,
                routing_reason=reason,
            )
        except Exception:
            logger.exception("Multi-agent workflow failed.")
            raise


agent_service = AgentService()
