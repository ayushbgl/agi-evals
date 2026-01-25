"""
Core abstract interfaces for the Game Arena platform.

This module provides the base classes that all games must implement
to integrate with the LLM evaluation arena.
"""

from core.game import Game
from core.state_adapter import StateAdapter
from core.action_parser import ActionParser, ActionParseError
from core.agent import Agent, AgentType

__all__ = [
    "Game",
    "StateAdapter",
    "ActionParser",
    "ActionParseError",
    "Agent",
    "AgentType",
]
