"""Configuration models for the Game Arena platform."""

from typing import Literal, Optional, List, Union, Dict, Any
from pydantic import BaseModel, Field, field_validator

from arena.registry import GAME_REGISTRY


class LLMConfig(BaseModel):
    """Configuration for LLM parameters."""

    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1)
    timeout: float = Field(default=60.0, ge=1.0, description="API timeout in seconds")

    class Config:
        extra = "forbid"


class PlayerConfig(BaseModel):
    """
    Configuration for a single player.

    This is a base configuration that works across games.
    Game-specific player configs can extend this.
    """

    id: str = Field(..., description="Unique player identifier")
    type: Literal["llm", "manual", "random", "rule_based"] = Field(
        ..., description="Type of agent controlling this player"
    )
    model: Optional[str] = Field(
        default=None,
        description="LLM model name (e.g., 'gpt-4o', 'claude-3-opus') for LLM players"
    )
    llm_config: Optional[LLMConfig] = Field(
        default=None,
        description="LLM-specific configuration"
    )

    # Game-specific fields (optional)
    team: Optional[str] = Field(default=None, description="Team assignment (for team games)")
    role: Optional[str] = Field(default=None, description="Role (for games with roles)")
    color: Optional[str] = Field(default=None, description="Color (for games with colors)")

    class Config:
        extra = "allow"  # Allow game-specific extra fields


class ArenaConfig(BaseModel):
    """
    Main configuration for Arena game execution.

    Supports multiple game types through the game_type field.
    """

    game_type: str = Field(
        ...,
        description="Type of game to run (e.g., 'catan', 'codenames')"
    )
    players: List[PlayerConfig] = Field(
        ...,
        min_length=2,
        description="List of player configurations"
    )
    game_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Game-specific configuration"
    )
    max_turns: int = Field(
        default=1000,
        ge=1,
        description="Maximum turns before game truncation"
    )

    # Storage
    log_dir: str = Field(
        default="./game_logs",
        description="Directory for game logs"
    )
    checkpoint_dir: str = Field(
        default="./checkpoints",
        description="Directory for checkpoints"
    )
    save_checkpoints: bool = Field(
        default=False,
        description="Whether to save checkpoints"
    )
    checkpoint_interval: int = Field(
        default=10,
        ge=1,
        description="Save checkpoint every N turns"
    )

    # Execution
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility"
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose logging"
    )

    @field_validator("game_type")
    @classmethod
    def validate_game_type(cls, v: str) -> str:
        """Validate that game_type is registered."""
        if v not in GAME_REGISTRY:
            available = list(GAME_REGISTRY.keys())
            raise ValueError(
                f"Unknown game type: {v}. Available: {available}"
            )
        return v

    def get_player_by_id(self, player_id: str) -> Optional[PlayerConfig]:
        """Get player configuration by ID."""
        for player in self.players:
            if player.id == player_id:
                return player
        return None

    def get_llm_players(self) -> List[PlayerConfig]:
        """Get all LLM players."""
        return [p for p in self.players if p.type == "llm"]

    def get_players_by_team(self, team: str) -> List[PlayerConfig]:
        """Get all players on a team."""
        return [p for p in self.players if p.team == team]

    def get_players_by_role(self, role: str) -> List[PlayerConfig]:
        """Get all players with a specific role."""
        return [p for p in self.players if p.role == role]

    class Config:
        extra = "forbid"


def load_config(filepath: str) -> ArenaConfig:
    """
    Load configuration from YAML or JSON file.

    Args:
        filepath: Path to config file

    Returns:
        ArenaConfig instance
    """
    import json
    from pathlib import Path

    path = Path(filepath)
    content = path.read_text()

    if path.suffix in ['.yaml', '.yml']:
        try:
            import yaml
            data = yaml.safe_load(content)
        except ImportError:
            raise ImportError("PyYAML required for YAML config files: pip install pyyaml")
    else:
        data = json.loads(content)

    return ArenaConfig(**data)


def create_config(
    game_type: str,
    player_configs: List[Dict[str, Any]],
    **kwargs,
) -> ArenaConfig:
    """
    Create an ArenaConfig from simple parameters.

    Args:
        game_type: Type of game
        player_configs: List of player configuration dicts
        **kwargs: Additional ArenaConfig fields

    Returns:
        ArenaConfig instance
    """
    players = [PlayerConfig(**pc) for pc in player_configs]
    return ArenaConfig(
        game_type=game_type,
        players=players,
        **kwargs,
    )
