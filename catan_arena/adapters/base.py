"""
Abstract base classes for game-LLM interface adapters.

These adapters handle the conversion between game state and LLM prompts,
and between LLM output and game actions.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class ActionParseError(Exception):
    """Raised when LLM output cannot be parsed into a valid action."""

    def __init__(self, message: str, raw_output: Optional[str] = None):
        super().__init__(message)
        self.raw_output = raw_output


class StateAdapter(ABC):
    """
    Abstract base class for converting game state to LLM prompts.

    Each game implementation should subclass this to provide game-specific
    prompt generation that converts complex game state into text/structured
    data that LLMs can understand and reason about.
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

        Args:
            public_state: Publicly visible game state (board, player summaries)
            private_state: Private state for current player (hand, hidden VPs)
            valid_actions: List of valid actions in structured format
            turn_history: Recent turn history for context

        Returns:
            Complete prompt string for the LLM
        """
        pass

    @abstractmethod
    def get_output_schema(self) -> Dict[str, Any]:
        """
        Return JSON schema for expected LLM output format.

        Returns:
            JSON Schema dict describing expected output structure
        """
        pass

    def format_system_prompt(self) -> str:
        """
        Return the system prompt for the LLM.

        Can be overridden by subclasses for game-specific instructions.
        """
        return "You are an expert game-playing AI."


class ActionParser(ABC):
    """
    Abstract base class for parsing LLM output into game actions.

    Each game implementation should subclass this to handle the specific
    action format and validation logic for that game.
    """

    @abstractmethod
    def parse(
        self,
        raw_output: str,
        valid_actions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Parse LLM output into a valid game action.

        Args:
            raw_output: Complete LLM response text
            valid_actions: List of valid actions for validation

        Returns:
            Parsed action dict matching the game's action format

        Raises:
            ActionParseError: If output cannot be parsed or action is invalid
        """
        pass

    def extract_reasoning(self, raw_output: str) -> str:
        """
        Extract the reasoning/chain-of-thought from LLM output.

        Args:
            raw_output: Complete LLM response text

        Returns:
            Extracted reasoning text, or empty string if not found
        """
        # Default implementation: everything before JSON block
        import re

        # Find JSON code block
        json_pattern = r'```(?:json)?\s*\{.*?\}\s*```'
        match = re.search(json_pattern, raw_output, re.DOTALL)

        if match:
            # Return everything before the JSON block
            reasoning = raw_output[:match.start()].strip()
            return reasoning

        # No JSON block found, return first part of output
        lines = raw_output.split('\n')
        reasoning_lines = []
        for line in lines:
            if line.strip().startswith('{'):
                break
            reasoning_lines.append(line)

        return '\n'.join(reasoning_lines).strip()
