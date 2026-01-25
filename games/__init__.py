"""
Game implementations for the Arena platform.

Each game is a submodule containing:
- game.py / environment.py: Core game logic
- state_adapter.py: State-to-prompt conversion
- action_parser.py: LLM output parsing
- prompts.py: Prompt templates
- config.py: Game-specific configuration

Supported games:
- catan: Settlers of Catan
- codenames: Codenames word game
"""

from typing import Dict, Any, Type
from core.game import Game

# Registry of available games
AVAILABLE_GAMES = ["catan", "codenames"]


def get_game_module(game_type: str):
    """
    Import and return the game module for the specified type.

    Args:
        game_type: Game identifier (e.g., "catan", "codenames")

    Returns:
        The game submodule

    Raises:
        ValueError: If game_type is not recognized
    """
    if game_type == "catan":
        from games import catan
        return catan
    elif game_type == "codenames":
        from games import codenames
        return codenames
    else:
        raise ValueError(f"Unknown game type: {game_type}. Available: {AVAILABLE_GAMES}")
