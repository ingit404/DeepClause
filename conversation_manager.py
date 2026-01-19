
from google.genai import types
from langchain_classic.memory import ConversationSummaryBufferMemory
from backend_gemini import client, rag_tool, GeminiLLM
from config import SYSTEM_INSTRUCTION
from google.genai import types
from langchain_classic.memory import ConversationSummaryBufferMemory
from backend_gemini import client, rag_tool, GeminiLLM
from config import SYSTEM_INSTRUCTION,MAX_OUTPUT_TOKENS,MODEL_NAME
import logging
from google.cloud import storage
from datetime import timedelta
from urllib.parse import urlparse

# Initialize Storage Client
storage_client = storage.Client()

def generate_signed_gcs_url(gcs_uri: str, expires_minutes: int = 15) -> str:
    """
    Converts gs://bucket/path/file.pdf
    → signed https://storage.googleapis.com/... link
    """
    try:
        parsed = urlparse(gcs_uri)
        bucket_name = parsed.netloc
        blob_name = parsed.path.lstrip("/")

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=expires_minutes),
            method="GET",
            response_disposition=f'attachment; filename="{blob.name.split("/")[-1]}"'
        )
        return url
    except Exception as e:
        logging.error(f"Error generating signed URL for {gcs_uri}: {e}")
        return None

# -------------------------
# LangChain memory management
# -------------------------

user_sessions = {}

def get_memory_for_session(session_id: str) -> ConversationSummaryBufferMemory:
    """Retrieves or creates a memory object for a given session ID."""
    if session_id not in user_sessions:
        user_sessions[session_id] = ConversationSummaryBufferMemory(
            llm=GeminiLLM(client=client),
            max_token_limit=1000,
            memory_key="history",
            input_key="input",
        )
    return user_sessions[session_id]

# -------------------------
# 6. Chat function (Streaming & Session-aware)
# -------------------------
def chat_stream(user_query: str, session_id: str):
    """
    Generator function that streams the response from Gemini.
    """
    memory = get_memory_for_session(session_id)
    history_text = memory.load_memory_variables({}).get("history", "")

    contents = []

    # Inject memory as conversational context
    if history_text.strip():
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        text=f"Conversation so far:\n{history_text}"
                    )
                ],
            )
        )

    # Current user message
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part(text=user_query)],
        )
    )

    # Gemini call with streaming

    response_stream = client.models.generate_content_stream(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=2048,
            tools=[rag_tool],
            system_instruction=SYSTEM_INSTRUCTION,
        ),
    )

    full_response_text = ""
    sources_blobs = []

    for chunk in response_stream:
        # 1. Extract text
        if chunk.text:
            text_fragment = chunk.text
            full_response_text += text_fragment
            yield text_fragment
        
        # 2. Extract grounding/sources
        # Candidates can be in the chunk
        if chunk.candidates:
             for candidate in chunk.candidates:
                if candidate.grounding_metadata and candidate.grounding_metadata.grounding_chunks:
                     for g_chunk in candidate.grounding_metadata.grounding_chunks:
                        if g_chunk.retrieved_context:
                             rc = g_chunk.retrieved_context
                             # Clean up URI for display (basename only)
                             display_name = rc.uri.split('/')[-1]
                             sources_blobs.append(f"- [📄 {display_name}]({rc.uri})\n  > {rc.text[:200].strip()}...")

    # Save complete interaction to memory
    memory.save_context(
        {"input": user_query},
        {"output": full_response_text},
    )

    # If we found sources, append them at the very end of the stream
    if sources_blobs:
        yield "\n\n**Sources:**\n" + "\n".join(set(sources_blobs))


# -------------------------
# 7. Synchronous Chat function (Returns Dictionary)
# -------------------------
def chat(user_query: str, session_id: str):
    """
    Synchronous function that returns a dictionary with answer and sources.
    """
    memory = get_memory_for_session(session_id)
    history_text = memory.load_memory_variables({}).get("history", "")

    contents = []

    # Inject memory as conversational context
    if history_text.strip():
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        text=f"Conversation so far:\n{history_text}"
                    )
                ],
            )
        )

    # Current user message
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part(text=user_query)],
        )
    )

    # Gemini call (No streaming)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.3,
            tools=[rag_tool],
            system_instruction=SYSTEM_INSTRUCTION,
        ),
    )

    # Save interaction to memory
    memory.save_context(
        {"input": user_query},
        {"output": response.text},
    )
    
    sources = []
    candidate = response.candidates[0]
    grounding = candidate.grounding_metadata

    if grounding and grounding.grounding_chunks:
        for chunk in grounding.grounding_chunks:
             if chunk.retrieved_context:
                rc = chunk.retrieved_context
                
                download_url = None
                if rc.uri.startswith("gs://"):
                    download_url = generate_signed_gcs_url(rc.uri)

                sources.append({
                    "file_name": rc.uri.split("/")[-1],
                    "download_url": download_url,
                    "text_snippet": rc.text[:300]
                })

    return {
        "answer": response.text,
        "sources": sources
    }
