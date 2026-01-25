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
]

GAME_TYPE = "codenames"
