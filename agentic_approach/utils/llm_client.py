from google import genai
from agentic_approach.config import PROJECT_ID, LOCATION

def get_gemini_client():
    """Returns a configured Gemini (Vertex AI) client."""
    return genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION
    )
