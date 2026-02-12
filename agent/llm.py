from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

from openai import OpenAI

ProviderName = Literal["openai", "openrouter"]


@dataclass(frozen=True)
class ProviderSpec:
    name: ProviderName
    api_key_env: str
    base_url: str | None
    default_model: str


PROVIDERS: dict[ProviderName, ProviderSpec] = {
    "openai": ProviderSpec(
        name="openai",
        api_key_env="OPENAI_API_KEY",
        base_url=None,
        default_model="gpt-4o-mini",
    ),
    "openrouter": ProviderSpec(
        name="openrouter",
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        # OpenRouter model IDs are typically like: "openai/gpt-4o-mini"
        default_model="openai/gpt-4o-mini",
    ),
}


def _coerce_provider(name: str | None) -> ProviderName:
    if not name:
        return "openai"
    name = name.strip().lower()
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {name}. Supported: {', '.join(PROVIDERS)}")
    return name  # type: ignore[return-value]


class LLMClient:
    """OpenAI-compatible chat completions client with pluggable providers.

    Supported providers (OpenAI-compatible):
      - openai     (OPENAI_API_KEY)
      - openrouter (OPENROUTER_API_KEY, base_url=https://openrouter.ai/api/v1)
    """

    def __init__(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        self.provider: ProviderName = _coerce_provider(provider or os.getenv("LLM_PROVIDER"))
        spec = PROVIDERS[self.provider]

        api_key = os.getenv(spec.api_key_env)
        if not api_key:
            raise RuntimeError(f"{spec.api_key_env} not set in environment")

        headers: dict[str, str] = {}
        if self.provider == "openrouter":
            # Optional but recommended by OpenRouter.
            site_url = os.getenv("OPENROUTER_SITE_URL")
            app_name = os.getenv("OPENROUTER_APP_NAME")
            if site_url:
                headers["HTTP-Referer"] = site_url
            if app_name:
                headers["X-Title"] = app_name

        self._client = OpenAI(
            api_key=api_key,
            base_url=spec.base_url,
            default_headers=headers or None,
        )

        self.model = (model or os.getenv("LLM_MODEL") or spec.default_model).strip()
        self.temperature = (
            float(os.getenv("LLM_TEMPERATURE"))
            if os.getenv("LLM_TEMPERATURE") is not None
            else (temperature if temperature is not None else 0.2)
        )
        self.max_tokens = (
            int(os.getenv("LLM_MAX_TOKENS"))
            if os.getenv("LLM_MAX_TOKENS") is not None
            else (max_tokens if max_tokens is not None else 900)
        )

    def configure(self, *, provider: str | None = None, model: str | None = None) -> None:
        """Switch active provider/model.

        Note: provider switch re-creates the underlying client (different API key/base_url).
        """
        new_provider = _coerce_provider(provider) if provider is not None else self.provider
        new_model = (model or self.model).strip()

        if new_provider == self.provider:
            self.model = new_model
            return

        # Re-init for provider change.
        self.__init__(provider=new_provider, model=new_model, temperature=self.temperature, max_tokens=self.max_tokens)

    def chat(self, *, system: str, user: str) -> str:
        """Send a system + user message and return the assistant reply."""
        response: Any = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""
