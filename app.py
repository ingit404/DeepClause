from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import os
from config import MODEL_NAME
import conversation_manager
# Initialize FastAPI
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    print(f"🔧 Active Model: {MODEL_NAME}")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Request Schema
class ChatRequest(BaseModel):
    message: str
    session_id: str

# Endpoints
@app.get("/")
async def read_root():
    return FileResponse("index.html")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    return conversation_manager.chat(request.message, request.session_id)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
