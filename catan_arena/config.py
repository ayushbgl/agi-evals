"""Configuration models for Catan-Arena."""

from typing import Literal, Optional, List
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for LLM parameters."""

    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1)
    timeout: float = Field(default=60.0, ge=1.0, description="API timeout in seconds")


class PlayerConfig(BaseModel):
    """Configuration for a single player."""

    id: str = Field(..., description="Unique player identifier")
    type: Literal["llm", "random", "minimax", "mcts", "value", "human"] = Field(
        ..., description="Type of player agent"
    )
    color: Literal["RED", "BLUE", "ORANGE", "WHITE"] = Field(
        ..., description="Player color"
    )
    model: Optional[str] = Field(
        default=None,
        description="LLM model name (e.g., 'claude-3-opus-20240229', 'gpt-4')"
    )
    llm_config: Optional[LLMConfig] = Field(
        default=None,
        description="LLM-specific configuration"
    )


class CatanGameConfig(BaseModel):
    """Configuration for Catan game parameters."""

    map_type: Literal["BASE", "MINI"] = Field(
        default="BASE", description="Map template to use"
    )
    vps_to_win: int = Field(
        default=10, ge=3, le=20, description="Victory points needed to win"
    )
    discard_limit: int = Field(
        default=7, ge=1, description="Cards above which player must discard on 7"
    )


class ArenaConfig(BaseModel):
    """Main configuration for Arena game execution."""

    game_type: Literal["catan"] = Field(default="catan")
    max_turns: int = Field(
        default=1000, ge=1, description="Maximum turns before game truncation"
    )
    players: List[PlayerConfig] = Field(
        ..., min_length=2, max_length=4, description="List of player configurations"
    )
    game_config: CatanGameConfig = Field(
        default_factory=CatanGameConfig, description="Game-specific configuration"
    )

    # Storage
    log_dir: str = Field(
        default="./game_logs", description="Directory for game logs"
    )
    checkpoint_dir: str = Field(
        default="./checkpoints", description="Directory for checkpoints"
    )
    save_checkpoints: bool = Field(
        default=True, description="Whether to save checkpoints"
    )
    checkpoint_interval: int = Field(
        default=10, ge=1, description="Save checkpoint every N turns"
    )

    # Execution
    seed: Optional[int] = Field(
        default=None, description="Random seed for reproducibility"
    )
    verbose: bool = Field(
        default=False, description="Enable verbose logging"
    )

    def get_player_by_color(self, color: str) -> Optional[PlayerConfig]:
        """Get player configuration by color."""
        for player in self.players:
            if player.color == color:
                return player
        return None


def load_config(filepath: str) -> ArenaConfig:
    """Load configuration from YAML or JSON file."""
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
