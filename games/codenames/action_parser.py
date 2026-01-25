"""
Codenames-specific action parser for converting LLM output to game actions.

Handles two types of actions:
- Spymaster clues: {"clue": "WORD", "number": N}
- Operative guesses: {"guess": "WORD"} or {"action": "PASS"}
"""

import json
import re
from typing import List, Dict, Any, Optional
import random

from core.action_parser import ActionParser, ActionParseError


class CodenamesActionParser(ActionParser):
    """
    Parses LLM output into Codenames game actions.

    Handles:
    1. Spymaster clues: {"clue": "word", "number": N}
    2. Operative guesses: {"guess": "word"}
    3. Pass actions: {"action": "PASS"}

    Supports flexible input formats including:
    - JSON in code blocks
    - Standalone JSON
    - Natural language fallback for simple cases
    """

    def __init__(
        self,
        fallback_to_random: bool = True,
        valid_words: Optional[List[str]] = None,
    ):
        """
        Initialize the parser.

        Args:
            fallback_to_random: If True, select random valid action on parse failure
            valid_words: List of valid words on the board (for validation)
        """
        super().__init__(fallback_to_random)
        self.valid_words = set(w.upper() for w in (valid_words or []))

    def set_valid_words(self, words: List[str]) -> None:
        """Update the list of valid words."""
        self.valid_words = set(w.upper() for w in words)

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
            Parsed action dict

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
                "Could not extract action from output",
                raw_output=raw_output[:500]
            )

        # Normalize and validate the parsed action
        normalized = self._normalize_action(parsed, valid_actions)

        if normalized is None:
            if self.fallback_to_random and valid_actions:
                return self.get_random_action(valid_actions)
            raise ActionParseError(
                f"Parsed action is not valid: {parsed}",
                raw_output=raw_output[:500]
            )

        return normalized

    def _normalize_action(
        self,
        parsed: Dict[str, Any],
        valid_actions: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Normalize and validate parsed action.

        Handles different field names and formats.
        """
        # Check for PASS action
        if parsed.get("action", "").upper() == "PASS":
            return {
                "action_type": "PASS",
                "reasoning": parsed.get("reasoning", ""),
            }

        # Check for clue (spymaster action)
        if "clue" in parsed:
            clue_word = str(parsed.get("clue", "")).strip().upper()
            clue_number = parsed.get("number", 0)

            # Validate number
            try:
                clue_number = int(clue_number)
            except (ValueError, TypeError):
                clue_number = 1

            clue_number = max(0, min(9, clue_number))

            return {
                "action_type": "GIVE_CLUE",
                "clue": clue_word,
                "number": clue_number,
                "reasoning": parsed.get("reasoning", ""),
            }

        # Check for guess (operative action)
        if "guess" in parsed:
            guess_word = str(parsed.get("guess", "")).strip().upper()

            # Validate word is in valid actions
            valid_words = [
                a.get("word", "").upper()
                for a in valid_actions
                if a.get("action_type") == "GUESS"
            ]

            if guess_word in valid_words or not valid_words:
                return {
                    "action_type": "GUESS",
                    "word": guess_word,
                    "confidence": parsed.get("confidence", "medium"),
                    "reasoning": parsed.get("reasoning", ""),
                }

            # Word not valid - try fuzzy matching
            closest = self._find_closest_word(guess_word, valid_words)
            if closest:
                return {
                    "action_type": "GUESS",
                    "word": closest,
                    "confidence": parsed.get("confidence", "medium"),
                    "reasoning": parsed.get("reasoning", ""),
                    "_original_guess": guess_word,
                    "_corrected": True,
                }

        # Check for word field directly (alternative format)
        if "word" in parsed and "action_type" not in parsed:
            word = str(parsed.get("word", "")).strip().upper()
            return {
                "action_type": "GUESS",
                "word": word,
                "confidence": parsed.get("confidence", "medium"),
                "reasoning": parsed.get("reasoning", ""),
            }

        return None

    def _parse_natural_language(
        self,
        text: str,
        valid_actions: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Attempt to parse natural language into an action.

        Fallback when JSON parsing fails.
        """
        text_lower = text.lower()

        # Check for pass
        if any(phrase in text_lower for phrase in [
            "i pass", "i'll pass", "pass my turn", "end my turn",
            "stop guessing", "that's enough"
        ]):
            return {"action": "PASS"}

        # Check for clue format (word + number)
        # Pattern: "clue is X for N" or "X, N" or similar
        clue_patterns = [
            r'(?:my )?clue (?:is |:)?\s*"?(\w+)"?\s*(?:for |,)\s*(\d+)',
            r"i(?:'ll)? give (?:the clue |)\"?(\w+)\"?\s*(?:for |,)\s*(\d+)",
            r'"(\w+)"\s*(?:for |,)\s*(\d+)',
            r'(\w+)\s*,\s*(\d+)$',  # At end of text
        ]

        for pattern in clue_patterns:
            match = re.search(pattern, text_lower)
            if match:
                return {
                    "clue": match.group(1).upper(),
                    "number": int(match.group(2)),
                }

        # Check for guess (single word in quotes or capitalized)
        # Pattern: "I guess X" or "my guess is X"
        guess_patterns = [
            r'(?:i |my )(?:guess|pick|choose|select)\s*(?:is |:)?\s*"?(\w+)"?',
            r'(?:i\'ll |i will |let\'s )(?:go with|try|guess)\s*"?(\w+)"?',
            r'touching\s*"?(\w+)"?',
        ]

        for pattern in guess_patterns:
            match = re.search(pattern, text_lower)
            if match:
                word = match.group(1).upper()
                # Validate against valid actions
                valid_words = [
                    a.get("word", "").upper()
                    for a in valid_actions
                    if a.get("action_type") == "GUESS"
                ]
                if word in valid_words:
                    return {"guess": word}

        # Last resort: look for any valid word mentioned
        valid_words = [
            a.get("word", "").upper()
            for a in valid_actions
            if a.get("action_type") == "GUESS"
        ]

        for word in valid_words:
            if word.lower() in text_lower:
                return {"guess": word}

        return None

    def _find_closest_word(
        self,
        word: str,
        valid_words: List[str],
    ) -> Optional[str]:
        """
        Find the closest matching word using simple heuristics.

        Handles common typos and similar spellings.
        """
        if not valid_words:
            return None

        word = word.upper()

        # Exact match
        if word in valid_words:
            return word

        # Check for substring match
        for valid in valid_words:
            if word in valid or valid in word:
                return valid

        # Check for similar start
        for valid in valid_words:
            if valid.startswith(word[:3]) or word.startswith(valid[:3]):
                return valid

        return None

    def parse_clue(self, raw_output: str) -> Dict[str, Any]:
        """
        Parse output specifically as a spymaster clue.

        Convenience method for clue-specific parsing.
        """
        return self.parse(raw_output, [{"action_type": "GIVE_CLUE"}])

    def parse_guess(
        self,
        raw_output: str,
        available_words: List[str],
    ) -> Dict[str, Any]:
        """
        Parse output specifically as an operative guess.

        Convenience method for guess-specific parsing.
        """
        valid_actions = [
            {"action_type": "GUESS", "word": w}
            for w in available_words
        ]
        valid_actions.append({"action_type": "PASS"})

        return self.parse(raw_output, valid_actions)
