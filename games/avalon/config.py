"""Avalon game configuration."""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from games.avalon.roles import Role, SPY_ROLES


class AvalonGameConfig(BaseModel):
    """Configuration for an Avalon game.

    Attributes:
        roles: Explicit role pool as a list of role-name strings.  When
               ``None`` the standard role table for the player count is used.
               The list length must equal the number of players configured in
               the arena.
    """

    roles: Optional[List[str]] = Field(
        default=None,
        description=(
            "Custom role pool (e.g. ['Merlin', 'Percival', 'Loyal Servant', "
            "'Morgana', 'Assassin']). None uses the default pool for the player count."
        ),
    )

    @field_validator("roles")
    @classmethod
    def validate_roles(cls, roles: Optional[List[str]]) -> Optional[List[str]]:
        if roles is None:
            return roles

        valid_names = {r.value for r in Role}
        for name in roles:
            if name not in valid_names:
                raise ValueError(
                    f"Unknown role: '{name}'. Valid roles: {sorted(valid_names)}"
                )

        n = len(roles)
        if not (5 <= n <= 10):
            raise ValueError("Role pool must contain 5–10 roles (one per player)")

        spy_count = sum(1 for r in roles if Role(r) in SPY_ROLES)
        if spy_count < 1:
            raise ValueError("Role pool must include at least one Spy")
        if spy_count >= n:
            raise ValueError("Role pool must include at least one Resistance member")

        return roles

    class Config:
        extra = "forbid"
