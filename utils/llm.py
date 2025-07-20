# utils/llm.py
import os
from openai import OpenAI

# load your key however you like (dotenv, env var, etc.)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def call_llm(prompt: str,
                   model: str = "gpt-4o-mini",
                   temperature: float = 0.0) -> str:
    # Note: .create() is synchronous, but that's fine in an async def
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return resp.choices[0].message.content.strip()
