import os
from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Note
    DEFAULT_LLM_PROVIDER: Literal["gemini", "openai", "anthropic", "ollama", "groq"] = "gemini"
    DEFAULT_LLM_MODEL: str = "gemini-3.5-flash-lite"
    LLM_TEMPERATURE: float = 0.2

    # Note
    GOOGLE_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Note
    MCP_TRANSPORT: Literal["sse", "stdio"] = "sse"
    MCP_SERVER_HOST: str = "127.0.0.1"
    MCP_SERVER_PORT: int = 8001
    MCP_SSE_ENDPOINT: str = "http://127.0.0.1:8001/sse"

    # Note
    HYBRID_RETRIEVAL_USE_MOCK: bool = True
    HYBRID_RETRIEVAL_API_BASE_URL: str = "http://127.0.0.1:8000/api/v1/retrieval"
    HYBRID_RETRIEVAL_API_KEY: Optional[str] = None

    # Note
    ORCHESTRATOR_API_HOST: str = "0.0.0.0"
    ORCHESTRATOR_API_PORT: int = 8080
    DIAGNOSIS_DB_PATH: str = "data/diagnosis_history.sqlite3"
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
