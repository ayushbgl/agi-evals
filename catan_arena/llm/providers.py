"""
LLM provider factory for Catan-Arena.

Supports multiple LLM providers via LangChain:
- Anthropic (Claude)
- OpenAI (GPT-4)
- Google (Gemini)
"""

from typing import Dict, Any, Optional, Union
from functools import lru_cache


# Type alias for LLM instances
BaseChatModel = Any  # Will be langchain_core.language_models.chat_models.BaseChatModel


def get_llm_for_model(
    model_name: str,
    config: Optional[Dict[str, Any]] = None,
) -> BaseChatModel:
    """
    Factory function to create an LLM instance for the given model.

    Args:
        model_name: Model identifier (e.g., "claude-3-opus-20240229", "gpt-4", "gemini-pro")
        config: Optional configuration dict with temperature, max_tokens, etc.

    Returns:
        LangChain chat model instance

    Raises:
        ValueError: If model provider cannot be determined
        ImportError: If required LangChain package is not installed
    """
    config = config or {}

    # Extract common config
    temperature = config.get("temperature", 0.7)
    max_tokens = config.get("max_tokens", 2048)
    timeout = config.get("timeout", 60.0)

    model_lower = model_name.lower()

    # Anthropic (Claude)
    if "claude" in model_lower or "anthropic" in model_lower:
        return _create_anthropic_llm(model_name, temperature, max_tokens, timeout)

    # OpenAI (GPT)
    if "gpt" in model_lower or "openai" in model_lower or model_lower.startswith("o1"):
        return _create_openai_llm(model_name, temperature, max_tokens, timeout)

    # Google (Gemini)
    if "gemini" in model_lower or "google" in model_lower:
        return _create_google_llm(model_name, temperature, max_tokens, timeout)

    # Default: try OpenAI-compatible
    raise ValueError(
        f"Cannot determine provider for model '{model_name}'. "
        f"Supported prefixes: claude, gpt, gemini"
    )


def _create_anthropic_llm(
    model_name: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> BaseChatModel:
    """Create Anthropic (Claude) LLM instance."""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        raise ImportError(
            "langchain-anthropic is required for Claude models. "
            "Install with: pip install langchain-anthropic"
        )

    return ChatAnthropic(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def _create_openai_llm(
    model_name: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> BaseChatModel:
    """Create OpenAI (GPT) LLM instance."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "langchain-openai is required for OpenAI models. "
            "Install with: pip install langchain-openai"
        )

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def _create_google_llm(
    model_name: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> BaseChatModel:
    """Create Google (Gemini) LLM instance."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise ImportError(
            "langchain-google-genai is required for Gemini models. "
            "Install with: pip install langchain-google-genai"
        )

    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        max_output_tokens=max_tokens,
        timeout=timeout,
    )


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a model.

    Args:
        model_name: Model identifier

    Returns:
        Dict with provider, context_window, etc.
    """
    model_lower = model_name.lower()

    if "claude" in model_lower:
        return {
            "provider": "anthropic",
            "context_window": 200000 if "3" in model_name else 100000,
            "supports_vision": "3" in model_name,
        }
    elif "gpt-4" in model_lower:
        return {
            "provider": "openai",
            "context_window": 128000 if "turbo" in model_lower or "o" in model_lower else 8192,
            "supports_vision": "vision" in model_lower or "o" in model_lower,
        }
    elif "gpt-3.5" in model_lower:
        return {
            "provider": "openai",
            "context_window": 16385,
            "supports_vision": False,
        }
    elif "gemini" in model_lower:
        return {
            "provider": "google",
            "context_window": 1000000 if "1.5" in model_name else 32000,
            "supports_vision": True,
        }

    return {
        "provider": "unknown",
        "context_window": 4096,
        "supports_vision": False,
    }


# Common model aliases for convenience
MODEL_ALIASES = {
    # Anthropic
    "claude-opus": "claude-3-opus-20240229",
    "claude-sonnet": "claude-3-5-sonnet-20241022",
    "claude-haiku": "claude-3-haiku-20240307",

    # OpenAI
    "gpt4": "gpt-4-turbo",
    "gpt4-turbo": "gpt-4-turbo",
    "gpt4o": "gpt-4o",
    "gpt4o-mini": "gpt-4o-mini",

    # Google
    "gemini": "gemini-pro",
    "gemini-pro": "gemini-1.5-pro",
    "gemini-flash": "gemini-1.5-flash",
}


def resolve_model_alias(model_name: str) -> str:
    """
    Resolve model alias to full model name.

    Args:
        model_name: Model name or alias

    Returns:
        Full model name
    """
    return MODEL_ALIASES.get(model_name.lower(), model_name)
