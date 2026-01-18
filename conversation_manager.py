
from google.genai import types
from langchain_classic.memory import ConversationSummaryBufferMemory
from backend_gemini import client, rag_tool, GeminiLLM
from config import SYSTEM_INSTRUCTION

# LangChain memory used to store conversation history

# Validated Imports
from google.genai import types
from langchain_classic.memory import ConversationSummaryBufferMemory
from backend_gemini import client, rag_tool, GeminiLLM
from config import SYSTEM_INSTRUCTION
import logging

# -------------------------
# 4. LangChain memory management
# -------------------------
# Dictionary to store memory for each session_id
# Format: { "session_id": ConversationSummaryBufferMemory object }
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
    # We DO NOT ask for tools in the streaming config to force text generation if possible,
    # but for RAG we need the tool.
    # Note: Vertex AI RAG often returns the *source* in the first chunk or metadata,
    # and the answer in subsequent chunks.
    response_stream = client.models.generate_content_stream(
        model="gemini-2.5-flash",
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
