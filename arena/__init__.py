"""
Game Arena - Game-agnostic LLM evaluation platform.

This module provides the orchestration layer for running LLM agents
against turn-based strategy games. It is designed to be game-agnostic,
supporting any game that implements the core.Game interface.

Components:
- orchestration: LangGraph-based game execution
- llm: LLM player implementations and providers
- storage: Game logging and replay
- registry: Game type registration
- config: Arena configuration

Supported games are registered in the registry module.
"""

from arena.registry import (
    GAME_REGISTRY,
    get_game_class,
    get_game_factory,
    get_state_adapter_class,
    get_action_parser_class,
    get_game_config_class,
    register_game,
)
from arena.config import ArenaConfig, PlayerConfig, LLMConfig

__all__ = [
    "GAME_REGISTRY",
    "get_game_class",
    "get_game_factory",
    "get_state_adapter_class",
    "get_action_parser_class",
    "get_game_config_class",
    "register_game",
    "ArenaConfig",
    "PlayerConfig",
    "LLMConfig",
]
