
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
from conversation_manager import chat
from config import MODEL_NAME

# Initialize FastAPI
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    # print(f"\nStarting Server...")
    print(f"🔧 Active Model: {MODEL_NAME}")
    # print(f"✅ System Ready\n")

# Enable CORS (for local development flexibility)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for logo
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Request Model
class ChatRequest(BaseModel):
    message: str
    session_id: str

# Endpoints
@app.get("/")
async def read_root():
    return FileResponse("index.html")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    return chat(request.message, request.session_id)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
