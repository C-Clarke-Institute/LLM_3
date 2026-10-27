from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
# embedding_model = OllamaEmbeddings(model="nomic-embed-text")
#pip install langchain-huggingface
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
loader = PyPDFLoader("hrdocument.pdf",mode="page")
all_splits = text_splitter.split_documents(loader.load())
vector_store = InMemoryVectorStore(embedding_model)
vector_store.add_documents(all_splits)

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

system_prompt = """
You are an HR Assistant with access to tools:
1.) retrieve_context: use this to retrieve context on HR knowledge

use the tools to provide accurate well-sourced answers about HR topics.
Be clear about what information comes from the document.
If you cant find the answer from the retrieve context say so, do not invent answers
"""
agent = create_agent(
    model=llm,
    tools=[retrieve_context],
    system_prompt=system_prompt
)

user_query = input("Enter your question : ")
result = agent.invoke({"messages": [{"role": "user", "content": user_query }]})
print(result)
