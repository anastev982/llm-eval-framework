# eval/lim_client.py
import os
from typing import List, Dict, Any
from openai import OpenAI  # type: ignore

# Load .env file (if it exists and dotenv is available)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass

# Initialize OpenAI client
client = OpenAI()

def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4o-mini",
    **kwargs: Any,
) -> str:
   
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        timeout=30,
        **kwargs,
    )
    return response.choices[0].message.content or ""
