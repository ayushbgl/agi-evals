"""
Game-agnostic orchestration for the Arena platform.

Provides LangGraph-based game execution that works with any
game implementing the core.Game interface.
"""

from arena.orchestration.state import ArenaGameState, create_initial_state
from arena.orchestration.runner import run_arena_game, run_simple_game

__all__ = [
    "ArenaGameState",
    "create_initial_state",
    "run_arena_game",
    "run_simple_game",
]
