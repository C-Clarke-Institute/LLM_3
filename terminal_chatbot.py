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

messages = [
    {"role" : "system", "content" : "You are a helpful AI Assistant"}
]

while True:
    user_prompt = input("USER : ")
    messages.append({"role" : "user" , "content" : user_prompt})

    resp = requests.post(OPENAI_API_URL
                         , headers=HEADERS
                         , json={"model" : "gpt-4o-mini"})

    ai_response = resp.json()["choices"][0]["messages"]["content"]
    print(f"AI Assistant : {ai_response}")

    messages.append({"role" : "assistant", "content" : ai_response})
