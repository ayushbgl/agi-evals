"""
LLM integration components for the Arena platform.

Provides:
- LLM providers (OpenAI, Anthropic, Google)
- LLMPlayer class for LLM-based agents
- ManualAgent for human testing
"""

from arena.llm.manual_agent import ManualAgent

__all__ = [
    "ManualAgent",
]
