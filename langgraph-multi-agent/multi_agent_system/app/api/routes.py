import logging

from fastapi import APIRouter, HTTPException, status

from app.models.request_models import UserQuestionRequest
from app.models.response_models import AgentResponse
from app.services.agent_service import agent_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Multi-Agent AI"])


@router.post(
    "/ask",
    response_model=AgentResponse,
    summary="Ask the multi-agent AI system",
)
async def ask_question(request: UserQuestionRequest) -> AgentResponse:
    """Route a question to the appropriate specialist and return its answer."""
    try:
        return await agent_service.ask(request.question)
    except ValueError:
        logger.exception("Invalid agent routing decision.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The question could not be routed safely.",
        )
    except Exception:
        logger.exception("Question processing failed.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI service could not complete the request. Please try again.",
        )
