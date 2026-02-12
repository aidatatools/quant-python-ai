import os

from openai import OpenAI


class LLMClient:
    """Thin wrapper around OpenAI chat completions."""

    def __init__(self, model: str = "gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment")
        self._client = OpenAI(api_key=api_key)
        self.model = model

    def chat(self, system: str, user: str) -> str:
        """Send a system + user message and return the assistant reply."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""
