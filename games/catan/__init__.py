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
    "create_game",
]

GAME_TYPE = "catan"


def create_game(arena_config):
    """
    Factory: create a CatanAECEnv from an ArenaConfig.

    Wraps the Catanatron game engine in a PettingZoo-style AEC
    environment compatible with the arena Game interface.
    """
    from catan_arena.envs.catan_pettingzoo import CatanAECEnv

    game_config = arena_config.game_config or {}
    env = CatanAECEnv(
        num_players=len(arena_config.players),
        map_type=game_config.get("map_type", "BASE"),
        vps_to_win=game_config.get("vps_to_win", 10),
        max_turns=arena_config.max_turns,
    )
    env.reset(seed=arena_config.seed)
    return env
