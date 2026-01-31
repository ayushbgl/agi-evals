"""
Game implementations for the Arena platform.

Each game is a self-contained submodule under games/<game_type>/ providing:
- game.py: Core game logic implementing core.Game
- state_adapter.py: State-to-prompt conversion (core.StateAdapter)
- action_parser.py: LLM output parsing (core.ActionParser)
- config.py: Game-specific configuration (Pydantic model)
- create_game(): Factory function for instantiation from ArenaConfig

To add a new game, create a games/<name>/ directory with these components
and register it in arena/registry.py.
"""

from importlib import import_module


def get_game_module(game_type: str):
    """
    Import and return the game module for the specified type.

    Uses the convention that each game lives in games/<game_type>/.

    Args:
        game_type: Game identifier (e.g., "catan", "codenames", "simple_card")

    Returns:
        The game submodule

    Raises:
        ValueError: If game module cannot be found
    """
    try:
        return import_module(f"games.{game_type}")
    except ImportError:
        raise ValueError(
            f"Unknown game type: {game_type}. "
            f"No module found at games.{game_type}"
        )
