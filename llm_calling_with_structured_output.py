from typing import List

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

class MovieReview(BaseModel):
    """Analysis of a movie review"""
    sentiment: str = Field(description="positive, negative or neutral")
    confidence: float = Field(description="0.0 to 1.0 confidence score")
    key_points: List[str] = Field(description="Main Points from the review")
    recommended: bool = Field(description="Would you recommend this movie, boolean value")

class CityInfo(BaseModel):
    name: str = Field("Name of the city")
    country: str = Field("Country where the city is located")
    famous_for : str = Field("For what this city is famous for")

def calling_llm_with_structured_output():

    llm = ChatOpenAI(model="gpt-4o-mini")

    structured_output_llm = llm.with_structured_output(MovieReview)

    prompt_text = """This movie was abosolutely fantastic!
     The acting was superb, especially shanudrie great job.
     The plot kept me on the edge of my seat. However,
     the ending felt but rushed, overall a good movie"""

    result: MovieReview = structured_output_llm.invoke(prompt_text)

    print(result.confidence)
    print(result.key_points)
    print(result.sentiment)

def using_lcel(city: str): #langchain expression language

    prompt = ChatPromptTemplate.from_template(
        "provide information about the city - {city}"
    )
    llm = ChatOpenAI(model="gpt-4o-mini")
    structured_output_llm = llm.with_structured_output(CityInfo)

    chain = prompt | structured_output_llm

    result: CityInfo = chain.invoke({"city" : city})
    print(result)

using_lcel("colombo")

# calling_llm_with_structured_output()