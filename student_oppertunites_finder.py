import json
import re
from datetime import datetime
from typing import Literal, get_args
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field, HttpUrl

load_dotenv()


SYSTEM_PROMPT = """
You are an AI Student Opportunity Finder for Sri Lankan students.

Your job is to find relevant, genuine opportunities using web search.

Supported opportunity  types :
- Scholorships
- Internships
- University programs
- Free Online courses

RULES - 
1. Always use web_search to find current opportunities.
2. Use get_current_datetime for questions involving today, deadline relative to today,
or the current year.
3. Prefer official university, scholarship provider, employer, government, 
and course-provider websites.
4. Never invent an opportunity, organization, deadline, eligibility requirement,
fee, or application URL.
5. If a field cannot b verified, return null or explain that 
the information could not be confirmed.
6. Distinguish application deadline from course start dates.
7. Do not claim an opportunity is open unless the sources support that conclusion.
8. Consider the student's country, qualification, field, budget, and other preferences.
9. Include Only URLs present in actual search results.
10. USE LKR for Sri Lankan rupee amounts. 
Do not invent exchange rates or convert currencies without evidence.
11. Explain uncertainty in the confidence field.
12. Do not guarantee that a student is eligible. 
Explain that the student mus confirm requirements with the provide.
13. Return a concise, structured response.

"""

# Search tool

web_search = TavilySearch(
    max_results=8,
    topic="general",
    name="web_search",
    description=(
        "Search the wen for current scholarships,internships,  university programs,"
        "free courses, eligibility criteria, application deadlines, fees, and official application URL"
        "Provide a specific search query "
    ),
)

@tool
def get_current_datetime() -> str :
    """Get the current date and time in Sri Lanka"""
    now = datetime.now(ZoneInfo("Asia/Colombo"))
    return now.isoformat()


# Request Schema
OPPORTUNITY_KIND = Literal["scholarship","internship","university_program", "free_course"]

class OpportunityRequest(BaseModel) :

    student_name : str | None = None
    country : str = "Sri Lanka"
    qualification : str = Field(
        description="Education Level | example - A/L , diploma, undergraduate, graduate")
    field_of_interest : str  = Field(
        description= "Example : IT, AI, business, engineering")
    opportunity_type: list[OPPORTUNITY_KIND] = Field(
        default_factory=lambda : list(get_args(OPPORTUNITY_KIND)))
    preferred_location : str = "Any"
    budget : str = "Prefer free opportunities"
    deadline_preference : str = "Currently open or accpting applications soon"
    additional_requirements : str = ""
    thread_id : str = "default"

#-----------------------------
# structured output schema

class Opportunity(BaseModel):
    title : str = Field(description= " Name of the opportunity")
    organization : str = Field(description="University, employer , or provider")
    opportunity_type : OPPORTUNITY_KIND
    summary : str = Field(description=" Brief explanation of the opportunity")
    eligibility : list[str] = Field(
        description= "Eligibility requirements supported by sources")
    deadline : str | None = Field(description="Application deadline or null if unverified")
    deadline_status : Literal["open", "upcoming", "closed", "unknown"] = "unknown"
    cost : str | None = Field(description="Fee, stipend, or funding details if verified")
    currency : str | None = Field(
        description="Currency Code,such as LKR, USD, or null")
    location : str | None = None
    application_url : HttpUrl | None = Field(
        description="Application page URL from search results"
    )
    source_urls : list[HttpUrl] = Field(
        description="URLs  supporting this opportunity"
    )
    confidence: Literal["high", "medium", "low"]

class OpportunitySearchResponse(BaseModel):
    answer: str
    opportunities : list[Opportunity]
    key_points : list[str]
    search_date  : str
    limitations: list[str]

#agent
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
checkpointer = InMemorySaver()

agent = create_agent(model=model,
                     system_prompt=SYSTEM_PROMPT,
                     tools=[
                         web_search,get_current_datetime
                     ],
                     response_format=ToolStrategy(OpportunitySearchResponse),
                     checkpointer=checkpointer)

def extract_urls(value) -> set[str] :
    """Extract URLs From tools results, including JSON strings"""

    urls = set()

    if isinstance(value,dict):
        for key,item in value.items():
            if key in {"url", "source_url"} and isinstance(item, str):
                if item.startswith("https://","http://"):
                    urls.add(item.rsplit(".,);]"))
                else:
                    urls.update(extract_urls(item))

    elif isinstance(value,list):
        for item in value:
            urls.update(extract_urls(item))

    elif isinstance(value,str):

        try:
            parsed = json.loads(value)
            if isinstance(parsed,(dict,list)):
                urls.update(extract_urls(parsed))
        except (json.JSONDecodeError, TypeError):
            for match in re.findall(r'https?://[^\s"<>]+',value):
                urls.add(match.rstrip(".,);]"))

    return urls

def search_urls(messages) -> set[str]:
    """Collect URLs actually by web search tools"""

    urls = set()

    for message in messages :
        if isinstance(message, ToolMessage):
            urls.update(extract_urls(message.content))
    return urls

def normalize_urls(url: str) -> str :
    """Normalize Minor URL differences for comparison"""

    parts = urlsplit(url)
    return (f"{parts.scheme.lower()}://"
            f"{parts.netloc.lower()}"
            f"{parts.path.rstrip('/')}"
            f"{'?'+parts.query if parts.query else ''}")

def validate_sources(response : OpportunitySearchResponse, searched_urls : set[str]
                     ) ->OpportunitySearchResponse:
    """Remove source and application URLs not present in the actual search tool results"""

    verified = {normalize_urls(url) for url in searched_urls}

    for opportunity in response.opportunities:
        opportunity.source_urls = [url for url in searched_urls
                                   if normalize_urls(str(url)) in verified]

        application_url = opportunity.application_url

        if (application_url is not  None and normalize_urls(str(application_url)) not in verified):
            opportunity.application_url = None

        if not opportunity.source_urls :
            opportunity.confidence = "low"

    return response
.

