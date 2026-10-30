import os
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from config.settings import settings
from src.common.logging import logger

class LLMFactory:
    """
    Factory to instantiate any LLM provider (Gemini, OpenAI, Anthropic, Ollama, Groq)
    conforming to LangChain's BaseChatModel interface.
    """

    @staticmethod
    def get_llm(
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> BaseChatModel:
        provider = (provider or settings.DEFAULT_LLM_PROVIDER).lower()
        model = model_name or settings.DEFAULT_LLM_MODEL
        temp = temperature if temperature is not None else settings.LLM_TEMPERATURE

        logger.info(f"Instantiating LLM: provider='{provider}', model='{model}', temperature={temp}")

        if provider == "gemini":
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                api_key = settings.GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    logger.warning("GOOGLE_API_KEY is not set. Ensure it is configured in .env")
                return ChatGoogleGenerativeAI(
                    model=model if "gemini" in model else "gemini-2.5-flash",
                    temperature=temp,
                    google_api_key=api_key,
                    convert_system_message_to_human=True
                )
            except ImportError:
                raise ImportError("langchain-google-genai is required for Gemini. Run: pip install langchain-google-genai")

        elif provider == "openai":
            try:
                from langchain_openai import ChatOpenAI
                api_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
                return ChatOpenAI(
                    model=model if not model.startswith("gemini-") else "gpt-4o-mini",
                    temperature=temp,
                    api_key=api_key
                )
            except ImportError:
                raise ImportError("langchain-openai is required for OpenAI. Run: pip install langchain-openai")

        elif provider == "anthropic":
            try:
                from langchain_anthropic import ChatAnthropic
                api_key = settings.ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY")
                return ChatAnthropic(
                    model=model if not model.startswith("gemini-") else "claude-3-5-sonnet-20241022",
                    temperature=temp,
                    api_key=api_key
                )
            except ImportError:
                raise ImportError("langchain-anthropic is required for Anthropic. Run: pip install langchain-anthropic")

        elif provider == "ollama":
            try:
                from langchain_community.chat_models import ChatOllama
                return ChatOllama(
                    model=model if not model.startswith("gemini-") else "llama3.2",
                    base_url=settings.OLLAMA_BASE_URL,
                    temperature=temp
                )
            except ImportError:
                raise ImportError("langchain-community is required for Ollama. Run: pip install langchain-community")

        elif provider == "groq":
            try:
                from langchain_openai import ChatOpenAI
                api_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
                return ChatOpenAI(
                    model=model if not model.startswith("gemini-") else "llama-3.1-70b-versatile",
                    temperature=temp,
                    base_url="https://api.groq.com/openai/v1",
                    api_key=api_key
                )
            except ImportError:
                raise ImportError("langchain-openai is required for Groq. Run: pip install langchain-openai")

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}. Supported: gemini, openai, anthropic, ollama, groq")
