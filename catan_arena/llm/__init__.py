"""LLM integration for Catan-Arena."""

from catan_arena.llm.providers import get_llm_for_model
from catan_arena.llm.player import LLMPlayer

__all__ = ["get_llm_for_model", "LLMPlayer"]
