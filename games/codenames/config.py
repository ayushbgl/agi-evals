"""Codenames-specific configuration."""

from typing import Literal, Optional, List
from pydantic import BaseModel, Field, field_validator


class CodenamesPlayerConfig(BaseModel):
    """Configuration for a single Codenames player."""

    id: str = Field(..., description="Unique player identifier")
    team: Literal["red", "blue"] = Field(..., description="Player's team")
    role: Literal["spymaster", "operative"] = Field(..., description="Player's role")
    type: Literal["llm", "manual", "random"] = Field(
        ..., description="Type of agent controlling this player"
    )
    model: Optional[str] = Field(
        default=None,
        description="LLM model name (e.g., 'gpt-4o', 'claude-3-opus') for LLM players"
    )
    llm_config: Optional[dict] = Field(
        default=None,
        description="LLM-specific configuration (temperature, max_tokens, etc.)"
    )

    class Config:
        extra = "forbid"


class CodenamesGameConfig(BaseModel):
    """Configuration for Codenames game parameters."""

    players: List[CodenamesPlayerConfig] = Field(
        ...,
        min_length=4,  # Minimum: 1 spymaster + 1 operative per team
        description="List of player configurations"
    )
    word_list: Literal["standard", "easy", "tech", "combined"] = Field(
        default="standard",
        description="Which word list to use"
    )
    max_turns: int = Field(
        default=50,
        ge=1,
        description="Maximum turns before game truncation"
    )
    starting_team: Optional[Literal["red", "blue"]] = Field(
        default=None,
        description="Which team goes first (random if not specified)"
    )

    @field_validator("players")
    @classmethod
    def validate_team_composition(cls, players: List[CodenamesPlayerConfig]):
        """Ensure both teams have at least one spymaster and one operative."""
        red_spymasters = [p for p in players if p.team == "red" and p.role == "spymaster"]
        red_operatives = [p for p in players if p.team == "red" and p.role == "operative"]
        blue_spymasters = [p for p in players if p.team == "blue" and p.role == "spymaster"]
        blue_operatives = [p for p in players if p.team == "blue" and p.role == "operative"]

        if len(red_spymasters) < 1:
            raise ValueError("Red team must have at least one spymaster")
        if len(red_operatives) < 1:
            raise ValueError("Red team must have at least one operative")
        if len(blue_spymasters) < 1:
            raise ValueError("Blue team must have at least one spymaster")
        if len(blue_operatives) < 1:
            raise ValueError("Blue team must have at least one operative")

        # Warn if multiple spymasters (unusual but allowed)
        if len(red_spymasters) > 1 or len(blue_spymasters) > 1:
            # Just a note - multiple spymasters would alternate
            pass

        return players

    def get_team_players(self, team: str) -> List[CodenamesPlayerConfig]:
        """Get all players on a team."""
        return [p for p in self.players if p.team == team]

    def get_spymaster(self, team: str) -> CodenamesPlayerConfig:
        """Get the spymaster for a team (first one if multiple)."""
        spymasters = [p for p in self.players if p.team == team and p.role == "spymaster"]
        return spymasters[0]

    def get_operatives(self, team: str) -> List[CodenamesPlayerConfig]:
        """Get all operatives for a team."""
        return [p for p in self.players if p.team == team and p.role == "operative"]

    class Config:
        extra = "forbid"
