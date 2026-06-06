"""Centralised, validated application configuration.

Uses pydantic-settings so every setting is type-checked and overridable via
environment variables or a local `.env` file. A single cached `Settings`
instance is exposed through `get_settings()` (dependency-injection friendly).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings loaded from env / `.env`.

    Field names map case-insensitively to environment variables, so
    `openai_api_key` is populated from `OPENAI_API_KEY`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Model providers -------------------------------------------------
    # Required: app fails fast at startup if this is missing.
    openai_api_key: str = Field(..., description="OpenAI API key (OPENAI_API_KEY).")
    # Optional: only needed when running Groq-backed models.
    groq_api_key: str | None = Field(
        default=None, description="Groq API key (GROQ_API_KEY)."
    )

    # --- LangSmith tracing / evals --------------------------------------
    langsmith_api_key: str | None = Field(
        default=None, description="LangSmith API key (LANGSMITH_API_KEY)."
    )
    langsmith_tracing: bool = Field(
        default=False, description="Enable LangSmith tracing (LANGSMITH_TRACING)."
    )
    langsmith_project: str = Field(
        default="evals", description="LangSmith project name (LANGSMITH_PROJECT)."
    )
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        description="LangSmith API endpoint (LANGSMITH_ENDPOINT).",
    )

    # --- Application defaults -------------------------------------------
    default_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return the cached, validated settings singleton.

    The `.env` file is read and validated exactly once; import and call this
    everywhere rather than reading `os.environ` directly.
    """
    return Settings()
