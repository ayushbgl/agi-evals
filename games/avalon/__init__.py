"""Avalon game module for the Arena platform.

Provides a full implementation of *The Resistance: Avalon* for 5–10 players,
including role assignment, team proposal, voting, missions, and the
assassination phase.

Components:
- game:           Core game engine (AvalonGame)
- roles:          Role definitions, visibility rules, and static tables
- state_adapter:  Converts game state to role-specific LLM prompts
- action_parser:  Parses LLM output into validated game actions
- config:         Game configuration (Pydantic)
- prompts:        Prompt templates
"""

from games.avalon.game import AvalonGame, Phase
from games.avalon.roles import Role, SPY_ROLES, RESISTANCE_ROLES
from games.avalon.config import AvalonGameConfig
from games.avalon.state_adapter import AvalonStateAdapter
from games.avalon.action_parser import AvalonActionParser
from games.avalon.baseline_agents import AvalonBaselineAgent

__all__ = [
    "AvalonGame",
    "Phase",
    "Role",
    "SPY_ROLES",
    "RESISTANCE_ROLES",
    "AvalonGameConfig",
    "AvalonStateAdapter",
    "AvalonActionParser",
    "create_game",
    "AvalonBaselineAgent",
    "create_baseline_agent",
]

GAME_TYPE = "avalon"


def create_baseline_agent(agent_id, seed=None):
    """Factory: create an AvalonBaselineAgent."""
    return AvalonBaselineAgent(agent_id=agent_id, seed=seed)


def create_game(arena_config):
    """
    Factory: create an AvalonGame from an ArenaConfig.

    Player IDs are drawn from ``arena_config.players``.  An optional
    ``roles`` list may be passed via ``arena_config.game_config``.
    """
    players = [p.id for p in arena_config.players]
    game_config = arena_config.game_config or {}
    return AvalonGame(
        players=players,
        seed=arena_config.seed,
        roles=game_config.get("roles"),
    )
