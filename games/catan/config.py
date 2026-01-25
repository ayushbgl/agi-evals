"""Catan-specific configuration."""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class CatanGameConfig(BaseModel):
    """Configuration for Catan game parameters."""

    map_type: Literal["BASE", "MINI"] = Field(
        default="BASE",
        description="Map template to use"
    )
    vps_to_win: int = Field(
        default=10,
        ge=3,
        le=20,
        description="Victory points needed to win"
    )
    discard_limit: int = Field(
        default=7,
        ge=1,
        description="Cards above which player must discard on 7"
    )
    max_turns: int = Field(
        default=1000,
        ge=1,
        description="Maximum turns before game truncation"
    )

    class Config:
        extra = "forbid"
