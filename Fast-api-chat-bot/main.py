from fastapi import FastAPI
from chat_routes import router as chat_router

app = FastAPI(
    title= "AI Chatbot API",
    description="Simple AI Chatbot using FastAPI",
    version="1.0.0"
)

app.include_router(chat_router)

@app.get("/")
def root():
    return {"message":"AI chatbot API  is running"}

@app.get("/health")
def health():
    return {
        "status": "ok"
    }