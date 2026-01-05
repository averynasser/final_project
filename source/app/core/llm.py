from typing import List, Dict
from openai import OpenAI
from .config import OPENAI_API_KEY, CHAT_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)


def chat_completion(
    system_prompt: str,
    messages: List[Dict],
    model: str = CHAT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> str:
    """
    Wrapper sederhana untuk panggilan chat completion.
    messages: list of {"role": "user"/"assistant", "content": "..."}
    """
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    resp = client.chat.completions.create(
        model=model,
        messages=full_messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return resp.choices[0].message.content
