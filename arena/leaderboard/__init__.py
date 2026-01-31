"""
AGI-Evals leaderboard — mock benchmark generator and Rich terminal renderer.

Usage
-----
    python -m arena.leaderboard          # render to terminal

Programmatic
------------
    from arena.leaderboard import generate_benchmark, render_leaderboard

    matches, entries = generate_benchmark()
    render_leaderboard(entries, matches)
"""

from arena.leaderboard.data import (
    generate_benchmark,
    LeaderboardEntry,
    MatchRecord,
    ModelProfile,
    GameTypeStats,
    H2HRecord,
    MODEL_CATALOG,
    GAME_TYPES,
)
from arena.leaderboard.display import render_leaderboard


def main() -> None:
    """Generate the benchmark dataset and render it."""
    matches, entries = generate_benchmark()
    render_leaderboard(entries, matches)


__all__ = [
    "generate_benchmark",
    "render_leaderboard",
    "main",
    "LeaderboardEntry",
    "MatchRecord",
    "ModelProfile",
    "GameTypeStats",
    "H2HRecord",
    "MODEL_CATALOG",
    "GAME_TYPES",
]
