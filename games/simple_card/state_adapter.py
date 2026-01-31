"""
State adapter for Simple Card (Number Battle) game.

Converts the game state into a clear, strategically-oriented prompt.
When it is player_1's turn, the prompt includes the opponent's played
card so the LLM can make an informed decision.
"""

from typing import Any, Dict, List, Optional

from core.state_adapter import StateAdapter


class SimpleCardStateAdapter(StateAdapter):
    """Converts Number Battle state into LLM-readable prompts."""

    def state_to_prompt(
        self,
        public_state: Dict[str, Any],
        private_state: Dict[str, Any],
        valid_actions: List[Dict[str, Any]],
        turn_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        lines = []
        lines.append("# Number Battle")
        lines.append("")
        lines.append(
            f"Round {public_state['current_round']} of {public_state['total_rounds']}"
        )
        lines.append("")

        # Scores
        lines.append("## Scores")
        for pid, score in public_state["scores"].items():
            marker = " (you)" if pid == private_state["player_id"] else ""
            lines.append(f"  {pid}{marker}: {score} point(s)")
        lines.append("")

        # Opponent's card if visible (player_1's turn)
        if "opponent_card" in private_state:
            lines.append("## Opponent's Card This Round")
            lines.append(
                f"Your opponent played: **{private_state['opponent_card']}**"
            )
            lines.append("")

        # Player's hand
        lines.append("## Your Hand")
        hand = private_state.get("hand", [])
        lines.append(f"Cards: {', '.join(str(c) for c in hand)}")
        lines.append("")

        # Round history
        history = public_state.get("round_history", [])
        if history:
            lines.append("## Round History")
            for r in history:
                p0c = r.get("player_0_card", "?")
                p1c = r.get("player_1_card", "?")
                winner = r.get("round_winner")
                outcome = f"{winner} wins" if winner else "Tie"
                lines.append(f"  Round {r['round']}: {p0c} vs {p1c} -> {outcome}")
            lines.append("")

        # Available moves
        lines.append("## Your Move")
        lines.append("Choose a card to play:")
        for action in valid_actions:
            lines.append(f"  - Card {action['card']}")
        lines.append("")
        lines.append("Respond with JSON:")
        lines.append("```json")
        lines.append(
            '{"action_type": "PLAY_CARD", "card": <number>, "reasoning": "..."}'
        )
        lines.append("```")

        return "\n".join(lines)

    def format_system_prompt(self, role: Optional[str] = None, **kwargs) -> str:
        return (
            "You are playing Number Battle, a simple card game. "
            "Each round you play one card from your hand. "
            "The higher card wins the round and earns a point. "
            "The player with the most points after all rounds wins. "
            "Play strategically to maximize your wins."
        )

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["action_type", "card"],
            "properties": {
                "action_type": {"type": "string", "enum": ["PLAY_CARD"]},
                "card": {
                    "type": "integer",
                    "description": "The card value to play from your hand",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Your strategic reasoning",
                },
            },
        }
