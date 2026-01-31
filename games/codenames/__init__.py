"""
Codenames game implementation for the Arena platform.

This module provides a full Codenames implementation with:
- 5x5 word grid
- Red vs Blue teams
- Spymaster and Operative roles
- Full game rules including assassin

Components:
- game: Core game logic and state management
- state_adapter: Converts game state to role-specific LLM prompts
- action_parser: Parses LLM output (clues and guesses)
- prompts: Prompt templates for spymasters and operatives
- config: Game configuration
- words: Word lists for the game
"""

from games.codenames.game import (
    CodenamesGame,
    Team,
    Phase,
    CardType,
    ClueResult,
    GuessResult,
)
from games.codenames.config import CodenamesGameConfig, CodenamesPlayerConfig
from games.codenames.state_adapter import CodenamesStateAdapter
from games.codenames.action_parser import CodenamesActionParser

__all__ = [
    "CodenamesGame",
    "Team",
    "Phase",
    "CardType",
    "ClueResult",
    "GuessResult",
    "CodenamesGameConfig",
    "CodenamesPlayerConfig",
    "CodenamesStateAdapter",
    "CodenamesActionParser",
    "create_game",
]

GAME_TYPE = "codenames"


def create_game(arena_config):
    """
    Factory: create a CodenamesGame from an ArenaConfig.

    Extracts team and role assignments from the player configurations
    to set up the Codenames game structure.
    """
    red_spymaster = None
    red_operatives = []
    blue_spymaster = None
    blue_operatives = []

    for p in arena_config.players:
        if p.team == "red":
            if p.role == "spymaster":
                red_spymaster = p.id
            else:
                red_operatives.append(p.id)
        elif p.team == "blue":
            if p.role == "spymaster":
                blue_spymaster = p.id
            else:
                blue_operatives.append(p.id)

    return CodenamesGame(
        red_spymaster=red_spymaster or "red_spymaster",
        red_operatives=red_operatives or ["red_operative"],
        blue_spymaster=blue_spymaster or "blue_spymaster",
        blue_operatives=blue_operatives or ["blue_operative"],
        seed=arena_config.seed,
    )
