"""
Catan-specific action parser for converting LLM output to game actions.

Parses LLM responses (JSON in code blocks, raw JSON, or natural language)
into valid Catanatron Action objects.
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
import random

from core.action_parser import ActionParser, ActionParseError


class CatanActionParser(ActionParser):
    """
    Parses LLM output into Catan game actions.

    Supports multiple output formats:
    1. Strict JSON in markdown code block (preferred)
    2. JSON without code block
    3. Natural language fallback (experimental)

    Also performs validation against the list of valid actions
    to ensure the parsed action is actually playable.
    """

    # Action types that don't require a value
    NULL_VALUE_ACTIONS = {
        "ROLL",
        "END_TURN",
        "BUY_DEVELOPMENT_CARD",
        "PLAY_KNIGHT_CARD",
        "PLAY_ROAD_BUILDING",
        "CANCEL_TRADE",
    }

    def __init__(self, fallback_to_random: bool = True):
        """
        Initialize the parser.

        Args:
            fallback_to_random: If True, return random valid action on parse failure
        """
        super().__init__(fallback_to_random)

    def parse(
        self,
        raw_output: str,
        valid_actions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Parse LLM output into a valid game action.

        Args:
            raw_output: Complete LLM response text
            valid_actions: List of valid actions for this turn

        Returns:
            Parsed action dict matching one of valid_actions

        Raises:
            ActionParseError: If output cannot be parsed and fallback is disabled
        """
        # Try JSON extraction first
        parsed = self.extract_json(raw_output)

        if parsed is None:
            # Try natural language parsing
            parsed = self._parse_natural_language(raw_output, valid_actions)

        if parsed is None:
            if self.fallback_to_random and valid_actions:
                return self.get_random_action(valid_actions)
            raise ActionParseError(
                f"Could not extract action from output",
                raw_output=raw_output[:500]
            )

        # Validate against valid_actions
        validated = self._validate_action(parsed, valid_actions)

        if validated is None:
            if self.fallback_to_random and valid_actions:
                return self.get_random_action(valid_actions)
            raise ActionParseError(
                f"Action {parsed} is not in valid_actions list",
                raw_output=raw_output[:500]
            )

        return validated

    def _parse_natural_language(
        self,
        text: str,
        valid_actions: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Attempt to parse natural language into an action.

        This is a fallback when JSON parsing fails. Uses keyword matching
        to identify the intended action.
        """
        text_lower = text.lower()

        # Roll dice
        if any(phrase in text_lower for phrase in ["roll the dice", "roll dice", "i'll roll", "i will roll"]):
            return {"action_type": "ROLL", "value": None}

        # End turn
        if any(phrase in text_lower for phrase in ["end my turn", "end turn", "pass", "done"]):
            return {"action_type": "END_TURN", "value": None}

        # Build road
        if "build" in text_lower and "road" in text_lower:
            edge_match = re.search(r'\[(\d+)\s*,\s*(\d+)\]', text)
            if edge_match:
                return {
                    "action_type": "BUILD_ROAD",
                    "value": [int(edge_match.group(1)), int(edge_match.group(2))]
                }

        # Build settlement
        if "build" in text_lower and "settlement" in text_lower:
            node_match = re.search(r'(?:node|at)\s*(\d+)', text_lower)
            if node_match:
                return {
                    "action_type": "BUILD_SETTLEMENT",
                    "value": int(node_match.group(1))
                }

        # Build city
        if "build" in text_lower and "city" in text_lower:
            node_match = re.search(r'(?:node|at)\s*(\d+)', text_lower)
            if node_match:
                return {
                    "action_type": "BUILD_CITY",
                    "value": int(node_match.group(1))
                }

        # Buy development card
        if "buy" in text_lower and ("dev" in text_lower or "development" in text_lower):
            return {"action_type": "BUY_DEVELOPMENT_CARD", "value": None}

        # Play knight
        if "play" in text_lower and "knight" in text_lower:
            return {"action_type": "PLAY_KNIGHT_CARD", "value": None}

        return None

    def _validate_action(
        self,
        parsed: Dict[str, Any],
        valid_actions: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Validate parsed action against list of valid actions.

        Returns the matching valid action or None if not found.
        """
        action_type = parsed.get("action_type")
        value = parsed.get("value")

        # Normalize value
        value = self._normalize_value(value)

        # First pass: exact match
        for valid in valid_actions:
            if valid.get("action_type") != action_type:
                continue

            valid_value = self._normalize_value(valid.get("value"))

            # For null-value actions
            if action_type in self.NULL_VALUE_ACTIONS:
                if valid_value is None:
                    return valid

            # For actions with values
            if self._values_match(value, valid_value):
                return valid

        # Second pass: fuzzy match for edges and coordinates
        for valid in valid_actions:
            if valid.get("action_type") != action_type:
                continue

            valid_value = self._normalize_value(valid.get("value"))

            if self._values_fuzzy_match(value, valid_value, action_type):
                return valid

        return None

    def _normalize_value(self, value: Any) -> Any:
        """Normalize value for comparison."""
        if value is None:
            return None
        if isinstance(value, list):
            return tuple(value) if len(value) <= 10 else value
        if isinstance(value, dict):
            return value
        return value

    def _values_match(self, parsed_value: Any, valid_value: Any) -> bool:
        """Check if parsed value matches valid action value."""
        if parsed_value is None and valid_value is None:
            return True

        if parsed_value == valid_value:
            return True

        # Handle list/tuple equivalence
        if isinstance(parsed_value, (list, tuple)) and isinstance(valid_value, (list, tuple)):
            return list(parsed_value) == list(valid_value)

        return False

    def _values_fuzzy_match(
        self,
        parsed_value: Any,
        valid_value: Any,
        action_type: str,
    ) -> bool:
        """
        Attempt fuzzy matching for edge cases.

        Handles:
        - Edge tuples where order doesn't matter (BUILD_ROAD)
        - Coordinate variations (MOVE_ROBBER)
        """
        if parsed_value is None or valid_value is None:
            return False

        # BUILD_ROAD: Edge order doesn't matter
        if action_type == "BUILD_ROAD":
            if isinstance(parsed_value, (list, tuple)) and len(parsed_value) == 2:
                if isinstance(valid_value, (list, tuple)) and len(valid_value) == 2:
                    parsed_sorted = tuple(sorted(parsed_value))
                    valid_sorted = tuple(sorted(valid_value))
                    return parsed_sorted == valid_sorted

        # MOVE_ROBBER: Coordinate matching
        if action_type == "MOVE_ROBBER":
            if isinstance(parsed_value, (list, tuple)):
                parsed_coord = None

                if len(parsed_value) >= 1:
                    if isinstance(parsed_value[0], (list, tuple)):
                        parsed_coord = tuple(parsed_value[0])
                    else:
                        if len(parsed_value) == 3:
                            parsed_coord = tuple(parsed_value)

                if isinstance(valid_value, (list, tuple)) and len(valid_value) >= 1:
                    valid_coord = None

                    if isinstance(valid_value[0], (list, tuple)):
                        valid_coord = tuple(valid_value[0])

                    if parsed_coord == valid_coord:
                        return True

        # MARITIME_TRADE: Check resource patterns
        if action_type == "MARITIME_TRADE":
            if isinstance(parsed_value, (list, tuple)) and isinstance(valid_value, (list, tuple)):
                if len(parsed_value) == len(valid_value):
                    return list(parsed_value) == list(valid_value)

        return False

    def parse_to_catan_action(
        self,
        raw_output: str,
        valid_actions: List[Dict[str, Any]],
        player_color: str,
    ) -> Tuple[Optional[Any], Dict[str, Any]]:
        """
        Parse LLM output and convert to Catanatron Action.

        Args:
            raw_output: LLM response text
            valid_actions: List of valid actions
            player_color: Color string for the player

        Returns:
            Tuple of (Catanatron Action or None, parse metadata dict)
        """
        from catanatron.models.enums import Action, ActionType
        from catanatron.models.player import Color

        metadata = {
            "raw_output": raw_output,
            "parsed_successfully": False,
            "parse_error": None,
            "reasoning": self.extract_reasoning(raw_output),
        }

        try:
            parsed = self.parse(raw_output, valid_actions)
            metadata["parsed_action"] = parsed
            metadata["parsed_successfully"] = True

            # Convert to Catanatron Action
            action_type = ActionType[parsed["action_type"]]
            color = Color[player_color]
            value = parsed.get("value")

            # Convert value format if needed
            if action_type == ActionType.BUILD_ROAD and isinstance(value, list):
                value = tuple(sorted(value))
            elif action_type == ActionType.MOVE_ROBBER and isinstance(value, list):
                if len(value) >= 1 and isinstance(value[0], list):
                    coord = tuple(value[0])
                    victim = Color[value[1]] if len(value) > 1 and value[1] else None
                    value = (coord, victim)

            action = Action(color, action_type, value)
            return action, metadata

        except ActionParseError as e:
            metadata["parse_error"] = str(e)
            return None, metadata
        except Exception as e:
            metadata["parse_error"] = f"Unexpected error: {str(e)}"
            return None, metadata
