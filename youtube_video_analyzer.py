from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import YoutubeLoader
from langchain_openai import ChatOpenAI

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

load_dotenv()
loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=iJiBJAJRRkk"
                                        ,add_video_info=False)
docs = loader.load()
transcript = docs[0].page_content

llm = ChatOpenAI(model="gpt-4o-mini")

analysis_chain = analysis_prompt | llm
result = analysis_chain.invoke({"transcript" : transcript})
print(result)