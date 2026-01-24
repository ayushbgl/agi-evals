"""
Catan-Arena: LLM Benchmark Platform for Settlers of Catan

A multi-agent benchmarking platform where Large Language Models compete
in the board game Settlers of Catan.
"""

__version__ = "0.1.0"

from catan_arena.config import ArenaConfig, PlayerConfig, LLMConfig

__all__ = [
    "ArenaConfig",
    "PlayerConfig",
    "LLMConfig",
    "__version__",
]
