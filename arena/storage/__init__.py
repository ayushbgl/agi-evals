"""
Storage components for game logs and replays.
"""

# Re-export from existing catan_arena storage if available
try:
    from catan_arena.storage.game_log import GameLogWriter
    __all__ = ["GameLogWriter"]
except ImportError:
    __all__ = []
