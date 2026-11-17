import logging

from langgraph.graph import END, START, StateGraph

from app.agents.car_analyzer_agent import process_car_question
from app.agents.crypto_agent import process_crypto_question
from app.agents.general_agent import process_general_question
from app.agents.router_agent import route_question
from app.graph.routing import select_agent
from app.graph.state import AgentState
from app.models.agent_models import AgentName

logger = logging.getLogger(__name__)


async def router_node(state: AgentState) -> dict:
    decision = await route_question(state["user_question"])
    return {
        "user_question": decision.user_question,
        "selected_agent": decision.selected_agent,
        "routing_reason": decision.routing_reason,
        "error": None,
    }


async def crypto_node(state: AgentState) -> dict:
    answer = await process_crypto_question(state["user_question"])
    return {"agent_response": answer}


async def car_analyzer_node(state: AgentState) -> dict:
    answer = await process_car_question(state["user_question"])
    return {"agent_response": answer}


async def general_node(state: AgentState) -> dict:
    answer = await process_general_question(state["user_question"])
    return {"agent_response": answer}


def build_workflow():
    """Build and compile the conditional multi-agent LangGraph workflow."""
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node(AgentName.CRYPTO.value, crypto_node)
    graph.add_node(AgentName.CAR_ANALYZER.value, car_analyzer_node)
    graph.add_node(AgentName.GENERAL.value, general_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        select_agent,
        {
            AgentName.CRYPTO.value: AgentName.CRYPTO.value,
            AgentName.CAR_ANALYZER.value: AgentName.CAR_ANALYZER.value,
            AgentName.GENERAL.value: AgentName.GENERAL.value,
        },
    )
    graph.add_edge(AgentName.CRYPTO.value, END)
    graph.add_edge(AgentName.CAR_ANALYZER.value, END)
    graph.add_edge(AgentName.GENERAL.value, END)
    return graph.compile()


workflow = build_workflow()
