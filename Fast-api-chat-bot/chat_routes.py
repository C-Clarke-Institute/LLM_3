from fastapi import APIRouter
from schemas.chat_request import ChatRequest
from chat_service import chat_with_ai, get_history, clear_history
from schemas.chat_response import ChatResponse


router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/",response_model=ChatResponse)
def chat(request: ChatRequest):

     ai_reponse = chat_with_ai(request.message)

     return ChatResponse(response=ai_reponse)

@router.get("/history")
def get_all_history():
     messages = get_history()
     return {"messages" : messages}

@router.delete("/history")
def clear_chat_history():

     clear_history()

     return {"messages" : "history is Cleared"}