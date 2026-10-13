from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


def chat(role: str, message: str):

    prompt = ChatPromptTemplate.from_template(
        "You are a {role}, Explain {message} in 3 sentences"
    )

    llm = ChatOllama(model="llama3.2:latest")

    msg = prompt.invoke({"role" : role, "message" : message })
    resp = llm.invoke(msg)

    print(resp)

chat("Lecturer", "Tell me about transformer layers in llm")
