import os

import requests

from models.chat_models import ChatMessage


from dotenv import load_dotenv

load_dotenv("../.env")

API_KEY = os.getenv("OPENAI_API_KEY")

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

HEADERS = {
    "Authorization" : f"Bearer {API_KEY}",
    "Content-Type" : "application/json"
}


#in memory
messages = [
    ChatMessage(
        role="system",
        content="you are a helpfull AI assistant"
    )
]

def chat_with_ai(user_message:str) -> str :
    messages.append(ChatMessage(role="user", content = user_message))

    message_data = [message.model_dump() for message in messages]

    response = requests.post(OPENAI_API_URL,headers=HEADERS,json={"model" : "gpt-4o-mini",
                                                                  "messages" : message_data})

    response.raise_for_status()

    data = response.json()

    ai_response = data["choices"][0]["message"]["content"]

    messages.append(ChatMessage(role="assistant", content=ai_response))

    return ai_response

def get_history():
    return messages

def clear_history():
    global messages

    messages = [
        ChatMessage(
            role="system",
            content="you are a  helpfull AI assistant"
        )
    ]

    return  messages








