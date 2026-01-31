"""
Game registry for the Arena platform.

Manages registration and lookup of game implementations,
adapters, and parsers for different game types.
"""

from typing import Dict, Type, Any, Optional, Callable
from importlib import import_module

from core.game import Game
from core.state_adapter import StateAdapter
from core.action_parser import ActionParser


# Registry of available games.
# Each entry maps game_type to component paths (resolved lazily at runtime).
GAME_REGISTRY: Dict[str, Dict[str, str]] = {
    "catan": {
        "game_module": "games.catan",
        "game_class": "catan_arena.envs.catan_pettingzoo.CatanAECEnv",
        "game_factory": "games.catan.create_game",
        "state_adapter_class": "games.catan.state_adapter.CatanStateAdapter",
        "action_parser_class": "games.catan.action_parser.CatanActionParser",
        "config_class": "games.catan.config.CatanGameConfig",
    },
    "codenames": {
        "game_module": "games.codenames",
        "game_class": "games.codenames.game.CodenamesGame",
        "game_factory": "games.codenames.create_game",
        "state_adapter_class": "games.codenames.state_adapter.CodenamesStateAdapter",
        "action_parser_class": "games.codenames.action_parser.CodenamesActionParser",
        "config_class": "games.codenames.config.CodenamesGameConfig",
    },
    "simple_card": {
        "game_module": "games.simple_card",
        "game_class": "games.simple_card.game.SimpleCardGame",
        "game_factory": "games.simple_card.create_game",
        "state_adapter_class": "games.simple_card.state_adapter.SimpleCardStateAdapter",
        "action_parser_class": "games.simple_card.action_parser.SimpleCardActionParser",
        "config_class": "games.simple_card.config.SimpleCardGameConfig",
    },
}


def _import_class(class_path: str) -> Type:
    """
    Import a class from its fully qualified path.

    Args:
        class_path: Dot-separated path like "module.submodule.ClassName"

    Returns:
        The class object

    Raises:
        ImportError: If module or class cannot be found
    """
    parts = class_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ImportError(f"Invalid class path: {class_path}")

    module_path, class_name = parts
    module = import_module(module_path)
    return getattr(module, class_name)


def _import_callable(callable_path: str) -> Callable:
    """
    Import a callable (function or class) from its fully qualified path.

    Args:
        callable_path: Dot-separated path like "module.submodule.function_name"

    Returns:
        The callable object
    """
    parts = callable_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ImportError(f"Invalid callable path: {callable_path}")

    module_path, name = parts
    module = import_module(module_path)
    return getattr(module, name)


def get_game_class(game_type: str) -> Type[Game]:
    """
    Get the Game class for a game type.

    Args:
        game_type: Game identifier (e.g., "catan", "codenames")

    Returns:
        The Game class

    Raises:
        ValueError: If game_type is not registered
    """
    if game_type not in GAME_REGISTRY:
        raise ValueError(
            f"Unknown game type: {game_type}. "
            f"Available: {list(GAME_REGISTRY.keys())}"
        )

    class_path = GAME_REGISTRY[game_type].get("game_class")
    if not class_path:
        raise ValueError(f"No game class defined for: {game_type}")

    return _import_class(class_path)


def get_game_factory(game_type: str) -> Callable:
    """
    Get the factory function for creating a game instance.

    The factory accepts an ArenaConfig and returns a fully initialized
    Game instance. This is the preferred entry point for game creation.

    Args:
        game_type: Game identifier

    Returns:
        Factory callable: (ArenaConfig) -> Game

    Raises:
        ValueError: If game_type is not registered or has no factory
    """
    if game_type not in GAME_REGISTRY:
        raise ValueError(
            f"Unknown game type: {game_type}. "
            f"Available: {list(GAME_REGISTRY.keys())}"
        )

    factory_path = GAME_REGISTRY[game_type].get("game_factory")
    if not factory_path:
        raise ValueError(f"No game factory defined for: {game_type}")

    return _import_callable(factory_path)


def get_state_adapter_class(game_type: str) -> Type[StateAdapter]:
    """
    Get the StateAdapter class for a game type.

    Args:
        game_type: Game identifier

    Returns:
        The StateAdapter class
    """
    if game_type not in GAME_REGISTRY:
        raise ValueError(f"Unknown game type: {game_type}")

    class_path = GAME_REGISTRY[game_type].get("state_adapter_class")
    if not class_path:
        raise ValueError(f"No state adapter defined for: {game_type}")

    return _import_class(class_path)


def get_action_parser_class(game_type: str) -> Type[ActionParser]:
    """
    Get the ActionParser class for a game type.

    Args:
        game_type: Game identifier

    Returns:
        The ActionParser class
    """
    if game_type not in GAME_REGISTRY:
        raise ValueError(f"Unknown game type: {game_type}")

    class_path = GAME_REGISTRY[game_type].get("action_parser_class")
    if not class_path:
        raise ValueError(f"No action parser defined for: {game_type}")

    return _import_class(class_path)


def get_game_config_class(game_type: str) -> Type:
    """
    Get the game-specific config class for a game type.

    Args:
        game_type: Game identifier

    Returns:
        The config class (Pydantic model)
    """
    if game_type not in GAME_REGISTRY:
        raise ValueError(f"Unknown game type: {game_type}")

    class_path = GAME_REGISTRY[game_type].get("config_class")
    if not class_path:
        raise ValueError(f"No config class defined for: {game_type}")

    return _import_class(class_path)


def get_game_module(game_type: str):
    """
    Get the game module for a game type.

    Args:
        game_type: Game identifier

    Returns:
        The game module
    """
    if game_type not in GAME_REGISTRY:
        raise ValueError(f"Unknown game type: {game_type}")

    module_path = GAME_REGISTRY[game_type].get("game_module")
    if not module_path:
        raise ValueError(f"No game module defined for: {game_type}")

    return import_module(module_path)


def register_game(
    game_type: str,
    game_class: Optional[str] = None,
    game_factory: Optional[str] = None,
    state_adapter_class: Optional[str] = None,
    action_parser_class: Optional[str] = None,
    config_class: Optional[str] = None,
    game_module: Optional[str] = None,
) -> None:
    """
    Register a new game type.

    Args:
        game_type: Unique identifier for the game
        game_class: Fully qualified path to Game class
        game_factory: Fully qualified path to factory function (ArenaConfig -> Game)
        state_adapter_class: Path to StateAdapter class
        action_parser_class: Path to ActionParser class
        config_class: Path to config class
        game_module: Path to game module
    """
    if game_type in GAME_REGISTRY:
        entry = GAME_REGISTRY[game_type]
    else:
        entry = {}
        GAME_REGISTRY[game_type] = entry

    if game_class:
        entry["game_class"] = game_class
    if game_factory:
        entry["game_factory"] = game_factory
    if state_adapter_class:
        entry["state_adapter_class"] = state_adapter_class
    if action_parser_class:
        entry["action_parser_class"] = action_parser_class
    if config_class:
        entry["config_class"] = config_class
    if game_module:
        entry["game_module"] = game_module


def list_games() -> list:
    """List all registered game types."""
    return list(GAME_REGISTRY.keys())


def is_game_registered(game_type: str) -> bool:
    """Check if a game type is registered."""
    return game_type in GAME_REGISTRY
