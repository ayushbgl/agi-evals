"""
Catan game implementation for the Arena platform.

This module provides Settlers of Catan integration using the
Catanatron game engine as the backend.

Components:
- environment: PettingZoo AEC wrapper for multi-agent play
- state_adapter: Converts game state to LLM prompts
- action_parser: Parses LLM output to game actions
- config: Catan-specific configuration
"""

from games.catan.config import CatanGameConfig
from games.catan.state_adapter import CatanStateAdapter
from games.catan.action_parser import CatanActionParser

__all__ = [
    "CatanGameConfig",
    "CatanStateAdapter",
    "CatanActionParser",
]

GAME_TYPE = "catan"
