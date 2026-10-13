from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import ToolRuntime,tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

load_dotenv()

USERS = {
    "1": {
        "name": "Amal",
        "home_city": "Colombo",
        "weekend_budget_lkr": 5000
    },
    "2": {
        "name": "Nimal",
        "home_city": "Kandy",
        "weekend_budget_lkr": 7500
    },
    "3": {
        "name": "Kasun",
        "home_city": "Galle",
        "weekend_budget_lkr": 6000
    },
    "4": {
        "name": "Tharindu",
        "home_city": "Negombo",
        "weekend_budget_lkr": 4500
    },
    "5": {
        "name": "Sahan",
        "home_city": "Jaffna",
        "weekend_budget_lkr": 8000
    }
}
WEATHER = {
    "Kandy": "Warm and cloudy, 25C, light rain likely",
    "Colombo": "Hot and humid, 30C, partly cloudy",
    "Galle": "Warm and breezy, 28C, sunny",
    "Jaffna": "Hot and dry, 32C, mostly sunny",
    "Nuwara Eliya": "Cool and misty, 18C, light rain possible"
}

TRAINS = {
    ("Colombo", "Kandy"): {
        "train": "Udarata Manike",
        "duration": "approx 2h 45m",
        "fare_lkr": 300
    },
    ("Colombo", "Galle"): {
        "train": "Ruhunu Kumari",
        "duration": "approx 2h 30m",
        "fare_lkr": 250
    },
    ("Colombo", "Jaffna"): {
        "train": "Yal Devi",
        "duration": "approx 6h 30m",
        "fare_lkr": 500
    },
    ("Colombo", "Badulla"): {
        "train": "Podi Menike",
        "duration": "approx 8h 30m",
        "fare_lkr": 450
    },
    ("Colombo", "Matara"): {
        "train": "Galu Kumari",
        "duration": "approx 2h 45m",
        "fare_lkr": 300
    }
}

ATTRACTIONS = {
    "Kandy": [
        {"name": "Temple of Tooth", "cost_lkr": 2000},
        {"name": "Peradeniya Botanical Garden", "cost_lkr": 1000}
    ],
    "Colombo": [
        {"name": "Colombo National Museum", "cost_lkr": 1500},
        {"name": "Viharamahadevi Park", "cost_lkr": 500}
    ],
    "Galle": [
        {"name": "Galle Fort", "cost_lkr": 1000},
        {"name": "Jungle Beach", "cost_lkr": 500}
    ],
    "Nuwara Eliya": [
        {"name": "Gregory Lake", "cost_lkr": 800},
        {"name": "Victoria Park", "cost_lkr": 500}
    ],
    "Jaffna": [
        {"name": "Jaffna Fort", "cost_lkr": 1000},
        {"name": "Nallur Kandaswamy Temple", "cost_lkr": 500}
    ]
}
@dataclass
class Context:
    user_id: str

@tool
def get_user_profile(runtime: ToolRuntime[Context]):
    """Get the Current users name, home city and travel budget information, Call this first for any trip requested"""
    user = USERS.get(runtime.context.user_id)
    if user is None:
        return "Unknown user. Ask the user for their home city and budget"
    return (f"Name : {user['name']}, home city: {user['home_city']} "
            f"Weekend budger : {user['weekend_budget_lkr']}")
@tool
def get_weather_for_location(city: str):
    """Get Current weather for a srilankan city"""
    return WEATHER.get(city, f"No weather data for - {city}")

@tool
def search_train(origin: str, destination: str):
    """Search train options between two srilankan cities, fares are one-way"""
    options = TRAINS.get((origin, destination))
    if not options:
        return f"No train data from {origin} to {destination}, suggest a bus instead"
    return str(options)

@tool
def get_attractions(city: str):
    """List things to do in a city with entrance cost"""
    return str(WEATHER.get(city, f"No attraction data for {city}"))


class TripPlan(BaseModel):
    destination: str = Field(description="City the person should visit")
    transport: str = Field(description="Chosen transport with fare, e.g"
                                       " 'udarata manike train, rs 900 one-way")
    weather_summary: str = Field(description="One line summary for the destination")
    estimated_total_cost_lkr: float = Field(description="Return transport + food + attractions, in lkr")
    within_budget: bool = Field(description="True only if estimated cost within users budget")
    itinerary: List[str] =  Field(min_length=1, description="Short day by day activities")
    packing_tips: str = Field(description="Packing tips based on the weather")

SYSTEM_PROMPT = """
You are a specialized Srilankan Travel Planning Assistant.
 Your primary goal is to help users plan weekend trips within 
 Sri Lanka by assessing 
 their preferences, home city, budget, transport options, weather, and attractions.
You have access to a specific set of tools to 
fetch dynamic user data, weather, train schedules, and city attractions.
plan a trip and give a structured out in provided format
"""

model = ChatOpenAI(model="gpt-5.4-2026-03-05", temperature=0, max_tokens=16000)

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_profile, get_attractions, get_weather_for_location, search_train],
    context_schema=Context,
    response_format=ToolStrategy(TripPlan),
    checkpointer=InMemorySaver()
)

def ask(question: str, user_id: str, thread_id: str):
    """Conversation history of agent is maintained by thread id"""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id" : thread_id}},
        context=Context(user_id=user_id)
    )
    print(result)
    return result['structured_response']

print(ask("plan a weekend trip for me", "1", "sss"))

