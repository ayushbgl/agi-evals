"""
Abstract ActionParser interface for parsing LLM output into game actions.

Action parsers handle the conversion between LLM text responses and
structured game actions that can be executed by the game engine.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import re
import json


class ActionParseError(Exception):
    """
    Raised when LLM output cannot be parsed into a valid action.

    Attributes:
        message: Human-readable error description
        raw_output: The original LLM output that failed to parse
    """

    def __init__(self, message: str, raw_output: Optional[str] = None):
        super().__init__(message)
        self.raw_output = raw_output


class ActionParser(ABC):
    """
    Abstract base class for parsing LLM output into game actions.

    Each game implementation should subclass this to handle the specific
    action format and validation logic for that game. The parser is
    responsible for:

    1. Extracting JSON/structured data from LLM responses
    2. Mapping parsed data to valid game actions
    3. Validating actions against the list of legal moves
    4. Providing fallback behavior for invalid outputs

    LLM outputs can be messy (extra text, malformed JSON, etc.), so
    parsers should be robust and attempt multiple extraction strategies.

    Example:
        class MyGameParser(ActionParser):
            def parse(self, raw_output, valid_actions):
                data = self._extract_json(raw_output)
                action = {"action_type": data["move"], "position": data["pos"]}
                if action not in valid_actions:
                    raise ActionParseError("Invalid move")
                return action
    """

    def __init__(self, fallback_to_random: bool = True):
        """
        Initialize the parser.

        Args:
            fallback_to_random: If True, select a random valid action
                when parsing fails. If False, raise ActionParseError.
        """
        self.fallback_to_random = fallback_to_random

    @abstractmethod
    def parse(
        self,
        raw_output: str,
        valid_actions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Parse LLM output into a valid game action.

        This method should attempt to extract a structured action from
        the LLM's response text, validate it against valid_actions,
        and return a properly formatted action dict.

        Args:
            raw_output: Complete LLM response text
            valid_actions: List of valid actions for validation

        Returns:
            Parsed action dict matching the game's action format

        Raises:
            ActionParseError: If output cannot be parsed and
                fallback_to_random is False
        """
        pass

    def extract_json(self, raw_output: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON object from LLM output.

        Attempts multiple strategies to find JSON:
        1. JSON in markdown code block (```json ... ```)
        2. JSON in generic code block (``` ... ```)
        3. Standalone JSON object
        4. First {...} block found

        Args:
            raw_output: Complete LLM response text

        Returns:
            Parsed JSON dict, or None if no valid JSON found
        """
        # Strategy 1: JSON in markdown code block
        json_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(json_block_pattern, raw_output, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Strategy 2: Standalone JSON with action_type field
        action_json_pattern = r'\{[^{}]*"action_type"[^{}]*\}'
        match = re.search(action_json_pattern, raw_output, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Strategy 3: Any JSON object (find matching braces)
        brace_start = raw_output.find('{')
        if brace_start != -1:
            depth = 0
            for i, char in enumerate(raw_output[brace_start:], brace_start):
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(raw_output[brace_start:i + 1])
                        except json.JSONDecodeError:
                            break

        return None

    def extract_reasoning(self, raw_output: str) -> str:
        """
        Extract the reasoning/chain-of-thought from LLM output.

        Reasoning is typically the text before the JSON response.
        This is useful for logging and debugging LLM decisions.

        Args:
            raw_output: Complete LLM response text

        Returns:
            Extracted reasoning text, or empty string if not found
        """
        # Find JSON code block
        json_pattern = r'```(?:json)?\s*\{.*?\}\s*```'
        match = re.search(json_pattern, raw_output, re.DOTALL)

        if match:
            # Return everything before the JSON block
            reasoning = raw_output[:match.start()].strip()
            return reasoning

        # No JSON block found, look for first brace
        brace_pos = raw_output.find('{')
        if brace_pos > 0:
            return raw_output[:brace_pos].strip()

        # No JSON found at all, return first part
        lines = raw_output.split('\n')
        reasoning_lines = []
        for line in lines:
            if line.strip().startswith('{'):
                break
            reasoning_lines.append(line)

        return '\n'.join(reasoning_lines).strip()

    def find_closest_action(
        self,
        parsed: Dict[str, Any],
        valid_actions: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Find the closest matching valid action to a parsed action.

        Useful when the LLM's output is close but not exact.
        Default implementation checks for exact action_type match.

        Args:
            parsed: Parsed action dict from LLM
            valid_actions: List of valid actions

        Returns:
            Matching valid action, or None if no close match
        """
        if not parsed or not valid_actions:
            return None

        parsed_type = parsed.get("action_type")
        if not parsed_type:
            return None

        # Find actions with matching type
        matching = [a for a in valid_actions if a.get("action_type") == parsed_type]

        if len(matching) == 1:
            return matching[0]

        # If multiple matches, try to match on value
        parsed_value = parsed.get("value")
        if parsed_value is not None:
            for action in matching:
                if action.get("value") == parsed_value:
                    return action

        # Return first match if any
        return matching[0] if matching else None

    def get_random_action(
        self,
        valid_actions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Select a random valid action as fallback.

        Args:
            valid_actions: List of valid actions

        Returns:
            Randomly selected action

        Raises:
            ActionParseError: If no valid actions available
        """
        import random

        if not valid_actions:
            raise ActionParseError("No valid actions available")

        return random.choice(valid_actions)
