import os

import requests
from dotenv import load_dotenv

load_dotenv()


API_KEY = os.getenv("OPENAI_API_KEY")

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

HEADERS = {
    "Authorization" : f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}


def chat():

    payload = {
        "model" : "gpt-4o-mini",
        "messages" : [
            {"role" : "system", "content" : "You are a sarcastic non helpful assistant"},
            {"role" : "user", "content" : "Tell me how a LLM works"}
        ],
        "temperature" : 0.0
    }

    resp = requests.post(OPENAI_API_URL, headers=HEADERS, json=payload)

    data = resp.json()

    print(data)

    print(data['choices'][0]['message']['content'])

chat()