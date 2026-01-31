"""
Simple Card (Number Battle) game implementation.

A two-player card game where players draw cards and play them in rounds.
Higher card wins each round. The player who wins the most rounds wins.

Round sequence:
  1. player_0 plays a card (committed without seeing opponent's choice)
  2. player_1 plays a card (can see player_0's card, creating asymmetry)
  3. Higher card scores a point; ties score nothing

After all cards are played, the player with more points wins.
Tied points result in a draw.

This game serves as a minimal reference implementation demonstrating
all required components: Game, StateAdapter, ActionParser, Config,
and the create_game() factory.
"""

from games.simple_card.game import SimpleCardGame
from games.simple_card.config import SimpleCardGameConfig
from games.simple_card.state_adapter import SimpleCardStateAdapter
from games.simple_card.action_parser import SimpleCardActionParser

__all__ = [
    "SimpleCardGame",
    "SimpleCardGameConfig",
    "SimpleCardStateAdapter",
    "SimpleCardActionParser",
    "create_game",
]

GAME_TYPE = "simple_card"


def create_game(arena_config):
    """
    Factory: create a SimpleCardGame from an ArenaConfig.

    Requires exactly 2 players. Passes cards_per_player and
    max_card_value from game_config if provided.
    """
    players = [p.id for p in arena_config.players]
    if len(players) != 2:
        raise ValueError(
            f"Number Battle requires exactly 2 players, got {len(players)}"
        )

    game_config = arena_config.game_config or {}
    return SimpleCardGame(
        players=players,
        seed=arena_config.seed,
        cards_per_player=game_config.get("cards_per_player", 3),
        max_card_value=game_config.get("max_card_value", 10),
    )
