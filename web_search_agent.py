from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

load_dotenv()

SYSTEM_PROMPT = """
You are a careful web research assistant for Sri lankan Students

Tools:
- get_current_datetime: Call this first when the question involves 'today', 'latest', 'now' or 'this year'
- web_search : search the web for up-to-date information

Rules:

- Always use web_search for facts that can change (news, prices, exchange rates, results, events)..
- Do not answer from memory when search results are available.
- If results disagree or are weak, say no and lower your confidence.
- Put only URLS that appeared in the search result in 'source'.
- Prices and amounts: state the currency, and mention LKR when relevant to srilanka
"""

web_search = TavilySearch(
    max_results=5,
    topic="general",
    name="web_search",
    description="Search the web for current information"
)

@tool
def get_current_datetime() -> str:
    """Get the current date and time in Sri Lanka"""
    now = datetime.now(ZoneInfo("Asia/Colombo"))
    return str(now)

@dataclass
class ResearchAnswer:
    answer: str
    key_points: list[str]
    sources: list[str]
    confidence: str

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
ollama_model = ChatOllama(model="llama3.2:latest")

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[web_search, get_current_datetime],
    response_format=ToolStrategy(ResearchAnswer),
    checkpointer=InMemorySaver()
)

def research(question: str, thread_id: str = "1") -> ResearchAnswer:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id" : thread_id}}
    )
    return result

class ResearchRequest(BaseModel):
    question: str
    thread_id: str

app = FastAPI()

@app.post("/research")
def run_research(request: ResearchRequest):
    return research(request.question, request.thread_id)