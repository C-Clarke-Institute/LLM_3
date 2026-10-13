import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def chat():

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role" : "system", "content" : "You are a helpful ai assistant"},
            {"role" : "user", "content" : "Tell me 5 facts about srilanka"}
        ]
    )


    print(resp.choices[0].message.content)

def temperature():
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful ai assistant"},
            {"role": "user", "content": "Give me a creative name for education institute"}
        ],
        temperature=0.0#Making the model output more deterministic
    )
    print(resp.choices[0].message.content)

temperature()

# chat()
