"""
Action parser for Simple Card (Number Battle) game.

Extracts a PLAY_CARD action from LLM output. Supports JSON extraction
as well as natural-language fallback patterns like "play the 5" or
"I'll go with card 3".
"""

import re
from typing import Any, Dict, List, Optional

from core.action_parser import ActionParser, ActionParseError


class SimpleCardActionParser(ActionParser):
    """Parses LLM output into Number Battle PLAY_CARD actions."""

    def parse(
        self,
        raw_output: str,
        valid_actions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        # Try JSON extraction first
        parsed = self.extract_json(raw_output)

        if parsed is None:
            parsed = self._parse_natural_language(raw_output, valid_actions)

        if parsed is None:
            if self.fallback_to_random and valid_actions:
                return self.get_random_action(valid_actions)
            raise ActionParseError(
                "Could not extract card choice", raw_output=raw_output[:500]
            )

        # Normalize to a card value
        card = parsed.get("card")
        if card is None:
            # Search all int values in the parsed dict
            for v in parsed.values():
                if isinstance(v, int):
                    card = v
                    break

        if card is None:
            if self.fallback_to_random and valid_actions:
                return self.get_random_action(valid_actions)
            raise ActionParseError(
                "No card value found in output", raw_output=raw_output[:500]
            )

        # Validate against valid actions
        card = int(card)
        for action in valid_actions:
            if action.get("card") == card:
                return action

        # Card not in valid actions — fall back
        if self.fallback_to_random and valid_actions:
            return self.get_random_action(valid_actions)
        raise ActionParseError(
            f"Card {card} is not a valid play",
            raw_output=raw_output[:500],
        )

    def _parse_natural_language(
        self,
        text: str,
        valid_actions: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Extract a card number from natural-language text."""
        valid_cards = {a["card"] for a in valid_actions if "card" in a}

        patterns = [
            r'play\s+(?:card\s+)?(\d+)',
            r'card\s+(\d+)',
            r'choose\s+(\d+)',
            r'pick\s+(\d+)',
            r'go\s+with\s+(?:card\s+)?(\d+)',
            r'\b(\d+)\b',  # bare number as last resort
        ]

        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                card = int(match.group(1))
                if card in valid_cards:
                    return {"action_type": "PLAY_CARD", "card": card}

        return None
