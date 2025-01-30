import os
from typing import Dict
from crewai import LLM
import time

MODELS = [
    "openai/gpt-4o-mini",  # Primary model
]

def get_llm_config(model_index=0):
    # Add delay between API calls to respect rate limits
    time.sleep(5)
    
    return LLM(
        model=MODELS[model_index % len(MODELS)],
        api_key=os.getenv('OPENROUTER_API_KEY'),
        temperature=0.7,
        max_tokens=4000,
        base_url="https://openrouter.ai/api/v1"
    )

def get_completion(prompt: str) -> str:
    config = get_llm_config()
    response = config.get_completion(
        messages=[{
            "role": "system",
            "content": "You are a specialized SVG design assistant. Create SVG testimonials that are modern, responsive, and follow best practices."
        }, {
            "role": "user",
            "content": prompt
        }],
    )
    return response.choices[0].message.content
