
from dotenv import load_dotenv

from typing_extensions import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START, END


# ==========================================
# 1. INITIAL CONFIGURATION
# ==========================================

load_dotenv()

# Web search tool
web_search_tool = TavilySearch(max_results=3)

# Embedding model
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Pinecone vector database
vector_store = PineconeVectorStore(
    index_name="hrdocumentsv3",
    embedding=embedding_model,
    namespace="hr-document"
)

# LLM
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# ==========================================
# 2. DEFINE GRAPH STATE
# ==========================================

class RagState(TypedDict):
    question: str
    context: str
    is_relevant: bool
    answer: str
    enriched_question: str
    web_results: str


# ==========================================
# 3. QUERY ENHANCEMENT
# ==========================================

def enhance_query_with_llm(user_query: str) -> str:
    """Enhance user query for better semantic retrieval."""

    enhancement_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a query optimization expert for an HR knowledge base.

        Expand the query to improve document retrieval.

        Instructions:
        1. Expand abbreviations (PTO = Paid Time Off,
           HR = Human Resources).
        2. Include relevant HR keywords.
        3. Preserve the original question's intent.
        4. Return only the enhanced query.
        """),
        ("human", "{question}")
    ])

    enhancement_chain = enhancement_prompt | model

    try:
        response = enhancement_chain.invoke({
            "question": user_query
        })

        return response.content.strip()

    except Exception as e:
        print(f"Query enhancement error: {e}")
        return user_query


# ==========================================
# 4. RETRIEVE CONTEXT FROM PINECONE
# ==========================================

def retrieve_context(query: str, num_results: int = 3) -> str:
    """Retrieve relevant HR documents from Pinecone."""

    retrieved_docs = vector_store.similarity_search_with_score(
        query,
        k=num_results
    )

    if not retrieved_docs:
        return ""

    results = "\n\n".join(
        f"""
        [Similarity Score: {score}]
        Metadata: {doc.metadata}

        Content:
        {doc.page_content}
        """
        for doc, score in retrieved_docs
    )

    print("\n========== PINECONE RESULTS ==========")
    print(results)

    return results


# ==========================================
# 5. CONTEXT RELEVANCE VALIDATION
# ==========================================

def validate_context(question: str, context: str) -> bool:
    """Validate whether retrieved context can answer the question."""

    if not context.strip():
        return False

    validation_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a strict HR document relevance validator.

        Determine whether the provided context contains
        sufficient information to directly answer the question.

        Return only YES or NO.

        YES: The context contains sufficient relevant information.
        NO: The context is missing, irrelevant, or insufficient.
        """),

        ("human", """
        Question:
        {question}

        Retrieved Context:
        {context}
        """)
    ])

    validation_chain = validation_prompt | model

    try:
        result = validation_chain.invoke({
            "question": question,
            "context": context
        })

        return result.content.strip().upper() == "YES"

    except Exception as e:
        print(f"Context validation error: {e}")
        return False


# ==========================================
# 6. LANGGRAPH NODES
# ==========================================

# ---------- RETRIEVE NODE ----------

def retrieve_node(state: RagState):
    """Retrieve relevant context from Pinecone."""

    print("\n========== RETRIEVE NODE ==========")

    enriched_question = enhance_query_with_llm(
        state["question"]
    )

    print(f"Original Question: {state['question']}")
    print(f"Enhanced Question: {enriched_question}")

    context = retrieve_context(enriched_question)

    return {
        "context": context,
        "enriched_question": enriched_question
    }


# ---------- VALIDATE NODE ----------

def validate_node(state: RagState):
    """Validate retrieved context."""

    print("\n========== VALIDATE NODE ==========")

    is_relevant = validate_context(
        state["question"],
        state["context"]
    )

    print(f"Context Relevant: {is_relevant}")

    return {
        "is_relevant": is_relevant
    }


# ---------- CONDITIONAL ROUTING ----------

def decide_retrieval(state: RagState):
    """Decide between RAG and web search."""

    if state["is_relevant"]:
        print("\nRouting to RAG Answer Node")
        return "rag_answer"

    print("\nRouting to Web Search Node")
    return "web_search"


# ---------- RAG ANSWER NODE ----------

def rag_answer_node(state: RagState):
    """Generate answer using Pinecone context."""

    print("\n========== RAG ANSWER NODE ==========")

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are an HR assistant.

        Answer the user's question using ONLY the
        provided HR document context.

        Instructions:
        1. Do not use external knowledge.
        2. Do not make assumptions.
        3. If the answer is not available in the context,
           clearly mention that.
        4. Provide a clear and helpful answer.
        """),

        ("human", """
        Question:
        {question}

        HR Document Context:
        {context}
        """)
    ])

    rag_chain = rag_prompt | model

    response = rag_chain.invoke({
        "question": state["question"],
        "context": state["context"]
    })

    return {
        "answer": response.content
    }


# ---------- WEB SEARCH NODE ----------

def web_search_node(state: RagState):
    """Search the web when Pinecone context is insufficient."""

    print("\n========== WEB SEARCH NODE ==========")

    search_results = web_search_tool.invoke({
        "query": state["question"]
    })

    print("\nWeb Search Results:")
    print(search_results)

    return {
        "web_results": str(search_results)
    }


# ---------- WEB ANSWER NODE ----------

def web_answer_node(state: RagState):
    """Generate answer using web search results."""

    print("\n========== WEB ANSWER NODE ==========")

    web_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a helpful HR assistant.

        The internal HR knowledge base does not contain
        sufficient information to answer the question.

        Answer using the provided web search results.

        Instructions:
        1. Use the provided search results.
        2. Do not invent information.
        3. If information is insufficient, mention it.
        4. Clearly indicate that the answer is based
           on external web sources.
        5. Include source URLs when available in the results.
        """),

        ("human", """
        Question:
        {question}

        Web Search Results:
        {web_results}
        """)
    ])

    web_chain = web_prompt | model

    response = web_chain.invoke({
        "question": state["question"],
        "web_results": state["web_results"]
    })

    return {
        "answer": response.content
    }


# ==========================================
# 7. BUILD LANGGRAPH WORKFLOW
# ==========================================

workflow = StateGraph(RagState)

# Register nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("validate", validate_node)
workflow.add_node("rag_answer", rag_answer_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("web_answer", web_answer_node)

# Define starting point
workflow.add_edge(START, "retrieve")

# Retrieve -> Validate
workflow.add_edge("retrieve", "validate")

# Conditional routing
workflow.add_conditional_edges(
    "validate",
    decide_retrieval,
    {
        "rag_answer": "rag_answer",
        "web_search": "web_search"
    }
)

# RAG path
workflow.add_edge("rag_answer", END)

# Web search path
workflow.add_edge("web_search", "web_answer")

workflow.add_edge("web_answer", END)

# Compile graph
adaptive_rag = workflow.compile()


# ==========================================
# 8. EXECUTE ADAPTIVE RAG
# ==========================================

if __name__ == "__main__":

    question = input("\nAsk your HR question: ")

    initial_state = {
        "question": question,
        "context": "",
        "is_relevant": False,
        "answer": "",
        "enriched_question": "",
        "web_results": ""
    }

    try:
        result = adaptive_rag.invoke(initial_state)

        print("\n" + "=" * 50)
        print("FINAL ANSWER")
        print("=" * 50)

        print(result["answer"])

        print("\n" + "=" * 50)
        print("EXECUTION SUMMARY")
        print("=" * 50)

        if result["is_relevant"]:
            print("Source: Pinecone HR Knowledge Base")
        else:
            print("Source: External Web Search")

    except Exception as e:
        print(f"\nAdaptive RAG Error: {e}")
