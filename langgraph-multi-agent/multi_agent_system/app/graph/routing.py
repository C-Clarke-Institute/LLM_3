from app.models.agent_models import AgentName


def select_agent(state: dict) -> str:
    """Return the graph node matching the validated router decision."""
    selected = state.get("selected_agent")
    if isinstance(selected, AgentName):
        selected = selected.value
    allowed = {agent.value for agent in AgentName}
    if selected not in allowed:
        raise ValueError("Router returned an unsupported agent.")
    return selected
