from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import chatbot_ex

# 로그 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class ChatRequest(BaseModel):
    chatMessage: str


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        logger.info(f"Received request: {request.chatMessage}")
        chat_response = chatbot_ex.get_response(request.chatMessage)
        logger.info(f"Generated response: {chat_response}")
        output = {
            "status": 200,
            "message": "채팅 응답 성공",
            "body": {
                "chatMessage": chat_response["result"]
            }
        }
        return output
    except Exception as e:
        logger.error(f"Error during response generation: {e}")
        raise HTTPException(status_code=400, detail={
            "status": 400,
            "message": "채팅 응답 실패",
            "body": {
                "error": str(e)
            }
        })

@app.get("/")
async def root():
    return {"message": "chatbot"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
