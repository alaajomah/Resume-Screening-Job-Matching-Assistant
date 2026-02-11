# src/llm.py
import os, json
from typing import Any, Dict
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path



env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path, override=False)


def get_openai_client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

def call_llm_json(client: OpenAI, model: str, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return json.loads(resp.choices[0].message.content)
