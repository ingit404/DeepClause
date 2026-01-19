import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GOOGLE_GENAI_USE_VERTEXAI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

PROJECT_ID = os.getenv("PROJECT_ID", "lending-partner")
LOCATION = os.getenv("LOCATION", "asia-south1")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")

SYSTEM_INSTRUCTION = """
You are a Lending Compliance Assistant.

Rules:
- Use RBI policy documents for regulatory, eligibility, and rule-based questions.
- Use emails ONLY for recent approvals, rejections, follow-ups, or discussions.
- NEVER assume facts without retrieval.
- NEVER store email content in memory.
- If unsure, retrieve both policy documents and emails.
"""
