import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found in environment variables")

# Chat model configurations
CHAT_MODEL = "gpt-4o-mini"
DESCRIPTION_MODEL = "ft:gpt-4o-mini-2024-07-18:personal::AtrjXCvd"
SVG_MODEL = "ft:gpt-4o-mini-2024-07-18:personal::ArguXr7z"
MAX_TOKENS = 2000
TEMPERATURE = 0.7 