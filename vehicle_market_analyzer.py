from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
car_model = "axio"

urls = [
    f"https://patpat.lk/en/sri-lanka/vehicle/car/toyota/{car_model}",
    f"https://ikman.lk/en/ads?query={car_model}",
    f"https://riyasewana.com/search/{car_model}"
]

website_loader = WebBaseLoader(urls)
docs = website_loader.load()

summarizer_prompt = ChatPromptTemplate.from_template("""
Extract car price in rupees, manufacture year, mileage from provided 
website context

WebPages : 
{docs}
""")
summarizer_llm = ChatOpenAI(model="gpt-4o-mini")

summarizer_chain = summarizer_prompt | summarizer_llm
ad_summary = summarizer_chain.invoke(
    {"docs" : "\n".join([doc.page_content for doc in docs])}
)
analysis_prompt = ChatPromptTemplate.from_template(
    """
    You are an automotive assistant helping users find the best car deals
    Here is a summarized list of car ads for {car_model}
    
    Ads Summary: {ad_summary}
    """
)
analysis_llm = ChatOpenAI(model="gpt-5.4-nano-2026-03-17")
response_chain = analysis_prompt | analysis_llm
output = response_chain.invoke({"car_model" : car_model, "ad_summary": ad_summary})
print(output)
