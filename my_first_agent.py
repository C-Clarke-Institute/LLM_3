from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

@tool
def get_weather(city: str) -> str:
    """Search for information."""
    return f"Weather in {city} is sunny"


agent = create_agent(model="gpt-5.4-2026-03-05", tools=[get_weather])
result = agent.invoke({"messages": [{"role": "user", "content": "Whats the weather in"
                                                          "colombo"}]})
print(result)