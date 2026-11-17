import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

logger = logging.getLogger(__name__)


async def run_tool_enabled_agent(
    *,
    llm: Any,
    system_prompt: str,
    question: str,
    tools: list[Any],
    max_tool_rounds: int = 5,
) -> str:
    """Run a small asynchronous tool-calling loop using the supplied tools."""
    tool_map = {item.name: item for item in tools}
    model = llm.bind_tools(tools)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ]

    for _ in range(max_tool_rounds + 1):
        ai_message = await model.ainvoke(messages)
        messages.append(ai_message)
        calls = getattr(ai_message, "tool_calls", []) or []
        if not calls:
            content = ai_message.content
            if isinstance(content, str):
                return content
            return str(content)

        for call in calls:
            name = call.get("name")
            args = call.get("args", {})
            call_id = call.get("id", name or "unknown")
            selected_tool = tool_map.get(name)
            if selected_tool is None:
                result = f"Tool '{name}' is not available."
            else:
                try:
                    result = await selected_tool.ainvoke(args)
                except Exception as exc:
                    logger.exception("Tool execution failed: %s", name)
                    result = f"Tool execution failed ({type(exc).__name__})."
            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=call_id,
                )
            )

    # One final model call lets the agent summarize the tool results.
    final_message = await model.ainvoke(messages)
    return str(final_message.content)
