"""
Number Battle card game implementation.

Two players each receive cards from a shuffled deck. Players play one card
per round — the higher card wins the round and earns a point. The player
who wins the most rounds wins the game.

Round structure:
  1. player_0 plays a card (committed without seeing opponent's choice)
  2. player_1 plays a card (sees player_0's card, creating strategic asymmetry)
  3. Cards compared: higher card scores; ties score nothing
  4. Repeat until all cards are played

Deck: two copies of each value from 1 to max_card_value. Each player is
dealt cards_per_player cards. Total rounds equal cards_per_player.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

from core.game import Game


class SimpleCardGame(Game):
    """Two-player card game: play cards, higher card wins each round."""

    def __init__(
        self,
        players: List[str],
        seed: Optional[int] = None,
        cards_per_player: int = 3,
        max_card_value: int = 10,
    ):
        if len(players) != 2:
            raise ValueError("Number Battle requires exactly 2 players")
        self._players = list(players)
        self._seed = seed
        self._cards_per_player = cards_per_player
        self._max_card_value = max_card_value
        self.reset(seed)

    @property
    def game_type(self) -> str:
        return "simple_card"

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._seed = seed
        self._rng = random.Random(self._seed)

        # Deck: two copies of each value 1..max_card_value
        deck = list(range(1, self._max_card_value + 1)) * 2
        self._rng.shuffle(deck)

        # Deal cards_per_player cards to each player
        self._hands: Dict[str, List[int]] = {}
        idx = 0
        for pid in self._players:
            self._hands[pid] = sorted(deck[idx:idx + self._cards_per_player])
            idx += self._cards_per_player

        self._scores: Dict[str, int] = {pid: 0 for pid in self._players}
        self._current_round = 1
        self._total_rounds = self._cards_per_player

        # Phase tracks which player acts next within a round
        self._phase = "player_0_play"
        self._pending_card: Optional[int] = None  # player_0's card this round

        self._round_history: List[Dict[str, Any]] = []
        self._game_over = False
        self._winner: Optional[str] = None

    def get_players(self) -> List[str]:
        return list(self._players)

    def get_state(self) -> Dict[str, Any]:
        return {
            "public": self.get_public_state(),
            "private_states": {
                pid: self.get_private_state(pid) for pid in self._players
            },
            "metadata": {
                "current_round": self._current_round,
                "phase": self._phase,
                "game_over": self._game_over,
            },
        }

    def get_public_state(self) -> Dict[str, Any]:
        return {
            "scores": dict(self._scores),
            "current_round": self._current_round,
            "total_rounds": self._total_rounds,
            "round_history": list(self._round_history),
            "game_over": self._game_over,
            "winner": self._winner,
        }

    def get_private_state(self, player_id: str) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "player_id": player_id,
            "hand": list(self._hands.get(player_id, [])),
        }
        # player_1 sees player_0's pending card during their turn
        if player_id == self._players[1] and self._phase == "player_1_play":
            state["opponent_card"] = self._pending_card
        return state

    def get_current_player(self) -> str:
        if self._phase == "player_0_play":
            return self._players[0]
        elif self._phase == "player_1_play":
            return self._players[1]
        return ""

    def get_current_role(self) -> str:
        return "player"

    def get_available_actions(self) -> List[Dict[str, Any]]:
        if self._game_over:
            return []
        current = self.get_current_player()
        return [
            {"action_type": "PLAY_CARD", "card": card}
            for card in sorted(self._hands.get(current, []))
        ]

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        if action.get("action_type") != "PLAY_CARD":
            raise ValueError(f"Unknown action type: {action.get('action_type')}")

        card = action.get("card")
        current = self.get_current_player()

        if card not in self._hands.get(current, []):
            raise ValueError(f"Card {card} not in {current}'s hand")

        # Remove card from hand
        self._hands[current].remove(card)

        # player_0 plays first — store card and hand over to player_1
        if self._phase == "player_0_play":
            self._pending_card = card
            self._phase = "player_1_play"
            return {"played": card, "player": current}, self._game_over

        # player_1 played — resolve the round
        p0_card = self._pending_card
        p1_card = card

        result: Dict[str, Any] = {
            "round": self._current_round,
            "player_0_card": p0_card,
            "player_1_card": p1_card,
        }

        if p0_card > p1_card:
            self._scores[self._players[0]] += 1
            result["round_winner"] = self._players[0]
        elif p1_card > p0_card:
            self._scores[self._players[1]] += 1
            result["round_winner"] = self._players[1]
        else:
            result["round_winner"] = None  # tie

        self._round_history.append(result)
        self._pending_card = None

        # Advance to next round or end game
        if self._current_round >= self._total_rounds:
            self._game_over = True
            s0 = self._scores[self._players[0]]
            s1 = self._scores[self._players[1]]
            if s0 > s1:
                self._winner = self._players[0]
            elif s1 > s0:
                self._winner = self._players[1]
            # else: draw — _winner stays None
        else:
            self._current_round += 1
            self._phase = "player_0_play"

        return result, self._game_over

    def is_over(self) -> bool:
        return self._game_over

    def get_winner(self) -> Optional[str]:
        return self._winner

    def get_scores(self) -> Dict[str, Any]:
        return dict(self._scores)

    def serialize(self) -> Dict[str, Any]:
        return {
            "game_type": "simple_card",
            "seed": self._seed,
            "players": self._players,
            "cards_per_player": self._cards_per_player,
            "max_card_value": self._max_card_value,
            "hands": {pid: list(hand) for pid, hand in self._hands.items()},
            "scores": dict(self._scores),
            "current_round": self._current_round,
            "phase": self._phase,
            "pending_card": self._pending_card,
            "round_history": list(self._round_history),
            "game_over": self._game_over,
            "winner": self._winner,
        }
