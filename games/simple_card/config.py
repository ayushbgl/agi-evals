"""Simple Card (Number Battle) game configuration."""

from pydantic import BaseModel, Field


class SimpleCardGameConfig(BaseModel):
    """Configuration for Simple Card (Number Battle) game parameters."""

    cards_per_player: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of cards dealt to each player (also the number of rounds)",
    )
    max_card_value: int = Field(
        default=10,
        ge=2,
        le=100,
        description="Highest card value in the deck (deck contains two copies of 1..max)",
    )

    class Config:
        extra = "forbid"
