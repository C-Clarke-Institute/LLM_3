from datetime import datetime, timezone

from langchain_core.tools import tool


@tool
def get_current_utc_time() -> str:
    """Get the current UTC date and time."""
    return datetime.now(timezone.utc).isoformat()


GENERAL_TOOLS = [get_current_utc_time]
