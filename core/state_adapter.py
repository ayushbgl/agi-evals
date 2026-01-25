"""
Abstract StateAdapter interface for converting game state to LLM prompts.

State adapters handle the conversion between complex game state objects
and human-readable text prompts that LLMs can understand and reason about.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class StateAdapter(ABC):
    """
    Abstract base class for converting game state to LLM prompts.

    Each game implementation should subclass this to provide game-specific
    prompt generation. The adapter is responsible for:

    1. Converting complex game state into readable text
    2. Formatting valid actions in a clear structure
    3. Providing role-specific views (e.g., spymaster vs operative)
    4. Including relevant history for context

    The output should be designed to help LLMs make informed decisions
    while avoiding information leakage (e.g., not showing hidden cards).

    Example:
        class MyGameAdapter(StateAdapter):
            def state_to_prompt(self, public_state, private_state, valid_actions, turn_history):
                return f'''
                Current board: {self._format_board(public_state)}
                Your hand: {private_state['hand']}
                Available moves: {self._format_actions(valid_actions)}
                '''
    """

    @abstractmethod
    def state_to_prompt(
        self,
        public_state: Dict[str, Any],
        private_state: Dict[str, Any],
        valid_actions: List[Dict[str, Any]],
        turn_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Convert game state to an LLM prompt.

        This is the main method that generates the user message content
        for the LLM. It should present all relevant information clearly
        and guide the LLM toward making a valid decision.

        Args:
            public_state: Publicly visible game state (board, scores, etc.)
            private_state: Private state for current player (hand, role info)
            valid_actions: List of valid actions in structured format
            turn_history: Recent turn history for context (optional)

        Returns:
            Complete prompt string for the LLM user message
        """
        pass

    @abstractmethod
    def get_output_schema(self) -> Dict[str, Any]:
        """
        Return JSON schema for expected LLM output format.

        This schema defines the structure the LLM should use when
        responding. It helps with:
        - Instructing the LLM on expected format
        - Validating LLM responses
        - Structured output parsing

        Returns:
            JSON Schema dict describing expected output structure.
            Should include "type", "properties", and "required" fields.

        Example:
            return {
                "type": "object",
                "properties": {
                    "action_type": {"type": "string"},
                    "value": {"type": ["string", "number", "null"]},
                    "reasoning": {"type": "string"}
                },
                "required": ["action_type"]
            }
        """
        pass

    def format_system_prompt(self, role: Optional[str] = None) -> str:
        """
        Return the system prompt for the LLM.

        The system prompt sets up the LLM's persona and provides
        game rules and general instructions. Can be customized
        based on the player's role.

        Args:
            role: Optional role identifier for role-specific prompts

        Returns:
            System prompt string
        """
        return "You are an expert game-playing AI. Analyze the game state carefully and make strategic decisions."

    def format_valid_actions_for_prompt(
        self,
        valid_actions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Format valid actions for inclusion in the prompt.

        This method can simplify or restructure action representations
        to be more readable in prompts while preserving the information
        needed to select an action.

        Default implementation returns actions unchanged.

        Args:
            valid_actions: Raw valid actions from the game

        Returns:
            Formatted actions for prompt inclusion
        """
        return valid_actions

    def format_turn_history(
        self,
        turn_history: List[Dict[str, Any]],
        max_turns: int = 10,
    ) -> str:
        """
        Format recent turn history for the prompt.

        Provides context about recent game events to help the LLM
        understand the game flow and make informed decisions.

        Args:
            turn_history: List of turn records
            max_turns: Maximum number of recent turns to include

        Returns:
            Formatted history string
        """
        if not turn_history:
            return "No previous turns."

        recent = turn_history[-max_turns:]
        lines = []
        for turn in recent:
            player = turn.get("player_id", "Unknown")
            action = turn.get("action", {})
            action_type = action.get("action_type", "unknown")
            lines.append(f"- {player}: {action_type}")

        return "\n".join(lines)
