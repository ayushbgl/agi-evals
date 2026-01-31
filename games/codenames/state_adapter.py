"""
Codenames-specific state adapter for converting game state to LLM prompts.

Provides role-specific prompts:
- Spymaster: Sees all card colors, gives clues
- Operative: Sees only revealed colors, guesses based on clues
"""

from typing import List, Dict, Any, Optional

from core.state_adapter import StateAdapter
from games.codenames.prompts import (
    SPYMASTER_SYSTEM_PROMPT,
    OPERATIVE_SYSTEM_PROMPT,
    SPYMASTER_USER_PROMPT,
    OPERATIVE_USER_PROMPT,
    get_spymaster_system_prompt,
    get_operative_system_prompt,
)


class CodenamesStateAdapter(StateAdapter):
    """
    Converts Codenames game state to role-specific LLM prompts.

    Handles two distinct prompt types:
    1. Spymaster prompts: Show full key card, ask for clue
    2. Operative prompts: Show only revealed cards, ask for guess
    """

    def __init__(self):
        """Initialize the Codenames state adapter."""
        pass

    def state_to_prompt(
        self,
        public_state: Dict[str, Any],
        private_state: Dict[str, Any],
        valid_actions: List[Dict[str, Any]],
        turn_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Generate role-appropriate prompt for the LLM.

        Args:
            public_state: Publicly visible game state
            private_state: Player's private state (includes role and card_types for spymaster)
            valid_actions: Valid actions (not really used for Codenames)
            turn_history: Recent turn history

        Returns:
            Complete prompt string
        """
        role = private_state.get("role")
        team = private_state.get("team", "red")

        if role == "spymaster":
            return self._format_spymaster_prompt(public_state, private_state)
        else:
            return self._format_operative_prompt(public_state, private_state)

    def format_system_prompt(self, role: Optional[str] = None, **kwargs) -> str:
        """
        Return the system prompt for the LLM.

        Extracts team, score, and clue context from the public/private
        state dicts passed by the runner.

        Args:
            role: "spymaster" or "operative"
            **kwargs: Must include public_state and private_state dicts

        Returns:
            System prompt string
        """
        public_state = kwargs.get("public_state", {})
        private_state = kwargs.get("private_state", {})
        team = private_state.get("team", "red")
        team_color = team.upper()

        if role == "spymaster":
            if team_color == "RED":
                team_cards = public_state.get("red_remaining", 9)
                opponent_cards = public_state.get("blue_remaining", 8)
            else:
                team_cards = public_state.get("blue_remaining", 9)
                opponent_cards = public_state.get("red_remaining", 8)
            return get_spymaster_system_prompt(team, team_cards, opponent_cards)

        elif role == "operative":
            current_clue = public_state.get("current_clue") or {}
            clue = current_clue.get("word", "")
            number = current_clue.get("number", 0)
            return get_operative_system_prompt(team, clue, number)

        return "You are playing Codenames. Follow the instructions carefully."

    def _format_spymaster_prompt(
        self,
        public_state: Dict[str, Any],
        private_state: Dict[str, Any],
    ) -> str:
        """Format prompt for spymaster."""
        team = private_state.get("team", "red").upper()
        opponent = "BLUE" if team == "RED" else "RED"

        # Format the grid display
        grid_display = self._format_grid_for_display(
            public_state.get("grid", []),
            show_all=False  # Show words, revealed status
        )

        # Format key card (spymaster can see all colors)
        key_card = private_state.get("key_card", [])
        key_card_display = self._format_key_card(key_card)

        # Format revealed cards
        revealed_display = self._format_revealed(public_state)

        # Format clue history
        clue_history_display = self._format_clue_history(
            public_state.get("clue_history", [])
        )

        return SPYMASTER_USER_PROMPT.format(
            grid_display=grid_display,
            key_card_display=key_card_display,
            team_color=team,
            opponent_color=opponent,
            team_remaining=public_state.get(
                "red_remaining" if team == "RED" else "blue_remaining", 0
            ),
            opponent_remaining=public_state.get(
                "blue_remaining" if team == "RED" else "red_remaining", 0
            ),
            revealed_display=revealed_display,
            clue_history_display=clue_history_display,
        )

    def _format_operative_prompt(
        self,
        public_state: Dict[str, Any],
        private_state: Dict[str, Any],
    ) -> str:
        """Format prompt for operative."""
        team = private_state.get("team", "red").upper()
        opponent = "BLUE" if team == "RED" else "RED"

        # Get current clue
        current_clue = public_state.get("current_clue", {})
        clue_word = current_clue.get("word", "") if current_clue else ""
        clue_number = current_clue.get("number", 0) if current_clue else 0

        # Format the grid display (operative view - no hidden colors)
        grid_display = self._format_grid_for_operative(
            public_state.get("grid", [])
        )

        # Format available (unrevealed) words
        available_words = self._format_available_words(public_state.get("grid", []))

        # Recent guesses this round
        guess_history = public_state.get("guess_history", [])
        current_turn = public_state.get("turn_number", 1)
        guesses_this_round = [
            g for g in guess_history
            if g.get("turn") == current_turn and g.get("team") == team.lower()
        ]
        guesses_display = self._format_recent_guesses(guesses_this_round)

        # Format clue history
        clue_history_display = self._format_clue_history(
            public_state.get("clue_history", [])
        )

        return OPERATIVE_USER_PROMPT.format(
            grid_display=grid_display,
            team_color=team,
            opponent_color=opponent,
            team_remaining=public_state.get(
                "red_remaining" if team == "RED" else "blue_remaining", 0
            ),
            opponent_remaining=public_state.get(
                "blue_remaining" if team == "RED" else "red_remaining", 0
            ),
            clue=clue_word,
            number=clue_number,
            guesses_this_round=guesses_display,
            guesses_remaining=public_state.get("guesses_remaining", 0),
            clue_history_display=clue_history_display,
            available_words=available_words,
        )

    def _format_grid_for_display(
        self,
        grid: List[List[Dict]],
        show_all: bool = False,
    ) -> str:
        """Format the 5x5 grid for display."""
        lines = []
        lines.append("```")

        for row in grid:
            row_words = []
            for cell in row:
                word = cell.get("word", "???")
                revealed = cell.get("revealed", False)

                if revealed:
                    card_type = cell.get("card_type", "?")[0].upper()
                    row_words.append(f"[{word}]({card_type})")
                else:
                    row_words.append(f" {word} ")

            # Pad words for alignment
            padded = [w.center(16) for w in row_words]
            lines.append(" | ".join(padded))

        lines.append("```")
        return "\n".join(lines)

    def _format_grid_for_operative(self, grid: List[List[Dict]]) -> str:
        """Format grid for operative (shows revealed status only)."""
        lines = []
        lines.append("```")

        for row in grid:
            row_words = []
            for cell in row:
                word = cell.get("word", "???")
                revealed = cell.get("revealed", False)

                if revealed:
                    card_type = cell.get("card_type", "?")
                    # Use emoji/symbol for card type
                    if card_type == "red":
                        row_words.append(f"[{word}](RED)")
                    elif card_type == "blue":
                        row_words.append(f"[{word}](BLU)")
                    elif card_type == "bystander":
                        row_words.append(f"[{word}](---)")
                    elif card_type == "assassin":
                        row_words.append(f"[{word}](XXX)")
                    else:
                        row_words.append(f"[{word}](?)")
                else:
                    row_words.append(f" {word} ")

            padded = [w.center(18) for w in row_words]
            lines.append(" | ".join(padded))

        lines.append("```")
        lines.append("")
        lines.append("Legend: [WORD](RED/BLU/---/XXX) = revealed card type")
        return "\n".join(lines)

    def _format_key_card(self, key_card: List[List[str]]) -> str:
        """Format the key card (spymaster only)."""
        if not key_card:
            return "(Key card not available)"

        lines = ["```"]
        for row in key_card:
            lines.append("  ".join(row))
        lines.append("```")
        return "\n".join(lines)

    def _format_revealed(self, public_state: Dict) -> str:
        """Format list of revealed cards."""
        grid = public_state.get("grid", [])
        revealed = []

        for row in grid:
            for cell in row:
                if cell.get("revealed"):
                    word = cell.get("word", "?")
                    card_type = cell.get("card_type", "?").upper()
                    revealed.append(f"{word} ({card_type})")

        if not revealed:
            return "No cards revealed yet."

        return ", ".join(revealed)

    def _format_available_words(self, grid: List[List[Dict]]) -> str:
        """Format list of unrevealed words."""
        available = []
        for row in grid:
            for cell in row:
                if not cell.get("revealed"):
                    available.append(cell.get("word", "?"))

        if not available:
            return "All words have been revealed."

        # Format as comma-separated list
        return ", ".join(available)

    def _format_clue_history(self, clue_history: List[Dict]) -> str:
        """Format clue history."""
        if not clue_history:
            return "No clues given yet."

        lines = []
        for clue in clue_history:
            team = clue.get("team", "?").upper()
            word = clue.get("word", "?")
            number = clue.get("number", 0)
            lines.append(f"- {team}: \"{word}\" for {number}")

        return "\n".join(lines)

    def _format_recent_guesses(self, guesses: List[Dict]) -> str:
        """Format recent guesses this round."""
        if not guesses:
            return "No guesses yet this round."

        lines = []
        for g in guesses:
            word = g.get("word", "?")
            card_type = g.get("card_type", "?").upper()
            correct = "Correct!" if g.get("correct") else "Wrong"
            lines.append(f"- {word} -> {card_type} ({correct})")

        return "\n".join(lines)

    def get_output_schema(self) -> Dict[str, Any]:
        """JSON Schema for expected LLM output."""
        return {
            "oneOf": [
                {
                    "type": "object",
                    "description": "Spymaster clue",
                    "required": ["clue", "number"],
                    "properties": {
                        "clue": {"type": "string"},
                        "number": {"type": "integer", "minimum": 0, "maximum": 9},
                        "reasoning": {"type": "string"},
                    }
                },
                {
                    "type": "object",
                    "description": "Operative guess",
                    "required": ["guess"],
                    "properties": {
                        "guess": {"type": "string"},
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"]
                        },
                        "reasoning": {"type": "string"},
                    }
                },
                {
                    "type": "object",
                    "description": "Pass action",
                    "required": ["action"],
                    "properties": {
                        "action": {"type": "string", "enum": ["PASS"]},
                        "reasoning": {"type": "string"},
                    }
                }
            ]
        }

    def format_valid_actions_for_prompt(
        self,
        valid_actions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Format valid actions - for Codenames, just return as-is."""
        return valid_actions
