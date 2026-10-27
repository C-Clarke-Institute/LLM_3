import os

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pinecone import Pinecone, ServerlessSpec
from pydantic.v1 import BaseModel

load_dotenv()
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = PineconeVectorStore(
    index_name="hrdocumentsv3",
    embedding=embedding_model,
    namespace="hr-document"
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
ollama_model = ChatOllama(model="llama3.2:latest")

def enhance_query_with_llm(user_query: str) -> str:
    """enhance user query with llm for better semantic search"""
    enhancement_prompt = f"""
    You are a query optimization expert for an HR knowledge base.
    Expand the query to improve document retrieval

    Original query: {user_query}

    Instructions:
    1.) Expand abbreviations (PTO- paid time off, HR - Human resource)
    2.) Include relevant HR keywords
    3.) Return only the enhanced query (no explanations)
    """

    try:
        response = ollama_model.invoke(enhancement_prompt)
        return response.content.strip()
    except Exception as e:
        print(f"Exception occurred - {str(e)}")
        return user_query


@tool
def retrieve_context(query: str, num_results: int = 3) -> str:
    """Retrieve relavant context from HR documents in vector database"""
    enhanced_query = enhance_query_with_llm(query)
    retrieved_docs = vector_store.similarity_search_with_score(enhanced_query, k=num_results)

    results = "\n\n".join(f"[Score : {score}] {doc.metadata}\n {doc.page_content}"
                          for doc, score in retrieved_docs)
    print(f"Retrieved Results from vector store - {results}")
    return results

@tool
def write_results_to_file(content: str, filename: str = "query_results.txt"):
    """
    write query results to a text file for persistence storage

    :param content: The result content to write
    :param filename: Target filename (default: query_results.txt)
    :return: Completion Message
    """

    try:

        with open(filename, "a") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters"
    except Exception as e:
        return f"Error Writing -  {str(e)}"

system_prompt = """
You are an HR Assistant with access to tools:
1.) retrieve_context: use this to retrieve context on HR knowledge

use the tools to provide accurate well-sourced answers about HR topics.
Be clear about what information comes from the document.
If you cant find the answer from the retrieve context say so, do not invent answers
"""
agent = create_agent(
    model=llm,
    tools=[retrieve_context, write_results_to_file],
    system_prompt=system_prompt,
    middleware=[HumanInTheLoopMiddleware(
        interrupt_on={
            "retrieve_context": False,
            "write_results_to_file": True
        }
    )],
    checkpointer=InMemorySaver()
)
def approve_and_continue(thread_id: str, decision: str):
    try:
        agent_config = {"configurable" : {"thread_id" : thread_id}}
        agent_result = agent.invoke(
            Command(resume={"decisions": [{"type": decision}]}),
            config=agent_config
        )
        if "messages" in agent_result:
            return agent_result["messages"][-1].content
        else:
            return "No Result"

    except Exception as e:
        print(str(e))


# while True:
#     user_query = input("Enter your question : ")
#     config = {"configurable" : {"thread_id" : "thread_01"}}
#
#     result = agent.invoke({"messages":
#                                [{"role": "user", "content": user_query}]
#                            },
#                           config=config)
#     if "__interrupt__" in result:
#         print("Agent is Interrupted on write to file and waiting on human approval(approve/reject) : ")
#         decision = input("Enter your decision: ")
#         response = approve_and_continue("thread_01", decision)
#         print(response)
#
#     print(result)

# =========================================
#             Schemas
# =========================================

class QueryRequest(BaseModel):
    user_id : str
    question : str

class QueryResponse(BaseModel):
    question: str
    answer : str
    user_id : str
    message : str
    status : str

class ApproveRequest(BaseModel):
    user_id : str
    decision : str


app = FastAPI(title="HR Assistant RAG Agent")

@app.post("/approve")
def human_approval_requests(request : ApproveRequest):
    config = {"configurable" : {"thread_id" : request.user_id}}

    result = agent.invoke(
        Command(resume={"decisions" :[{"type" : request.decision}]}),
        config=config
    )

    answer = result["messages"][-1].content if result.got("messages") else "No Answer"

    return answer


@app.post("/query", response_model=QueryResponse)
def query(request:QueryRequest):
    config = {"configurable" : {"thread_id" : request.user_id}}
    resp = agent.invoke({"messages": [{"role": "user",
                                       "content" : request.question}]},
                        config=config)

    if '__interrupt__' in resp:
        return QueryResponse(
            question= request.question,
            answer="No Answer",
            user_id= request.user_id,
            message = "Write to file tool is pending human approval",
            status="Interrupted"
        )

    return QueryResponse(
        question=request.question,
        answer=resp["messages"][-1].content,
        user_id=request.user_id,
        message="Success",
        status="Success"
    )


