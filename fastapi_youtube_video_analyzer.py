from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException
load_dotenv()
class AnalysisRequest(BaseModel):
    url: str

class AnalysisResponse(BaseModel):
    topic: str = Field(description="Main Topic Covered in the video transcript")
    insights: List[str] = Field(description="3 positive insights from the video transcript")
    summary: str = Field(description="Summary of the video")
    audience: str = Field(description="Audience this video is recommended for")

llm = ChatOpenAI(model="gpt-4o-mini")

analysis_prompt = ChatPromptTemplate.from_template("""
You are an expert content analyst.
Analyze the following video transcript.

Extract:
- Main Topics covered
- Positive insights from the video
- summary
- who would benefit from watching

Transcript:

{transcript}
""")
async def get_transcript(url: str) -> str:
    try:
        loader = YoutubeLoader.from_youtube_url(url, add_video_info =False)
        docs = loader.load()
        if not docs:
            raise HTTPException(status_code=400, detail="no transcript found")
        return docs
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{str(e)}")

async def analyze_video (url: str) -> AnalysisResponse:
    transcript = await get_transcript(url)

    analysis_chain = analysis_prompt | llm.with_structured_output(AnalysisResponse)
    result = await analysis_chain.ainvoke({"transcript": transcript})
    return result

app = FastAPI(title="Youtube analyzer API")
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest):
    result = await analyze_video(request.url)
    return result

