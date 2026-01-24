"""
Game logging system for Catan-Arena.

Provides structured JSON logging of complete game sessions including:
- Game configuration and metadata
- Turn-by-turn state (public and private)
- LLM decision records with reasoning
- Final results and statistics
"""

import json
import gzip
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import uuid


SCHEMA_VERSION = "1.0.0"


@dataclass
class TurnRecord:
    """Record of a single turn in the game."""

    turn_number: int
    player_id: str
    phase: str
    public_state: Dict[str, Any]
    private_states: Dict[str, Dict[str, Any]]
    valid_actions: List[Dict[str, Any]]
    llm_decision: Optional[Dict[str, Any]]
    action: Dict[str, Any]
    action_result: Optional[Dict[str, Any]]
    timestamp: str


@dataclass
class GameResult:
    """Final result of a game."""

    termination_reason: str  # "victory", "turn_limit", "error"
    winner_id: Optional[str]
    final_scores: Dict[str, int]
    total_turns: int
    statistics: Dict[str, Any]


class GameLogWriter:
    """
    Writer for game log files.

    Creates structured JSON logs that capture the complete game session
    for replay and analysis.

    Example:
        writer = GameLogWriter(
            game_id="abc123",
            game_type="catan",
            players=[...],
            config={...}
        )

        # During game
        writer.record_turn(turn_data)

        # After game
        writer.finalize(result)
        writer.save("./logs/game_abc123.json")
    """

    def __init__(
        self,
        game_id: Optional[str] = None,
        game_type: str = "catan",
        players: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a new game log.

        Args:
            game_id: Unique game identifier (auto-generated if not provided)
            game_type: Type of game being played
            players: List of player configurations
            config: Game configuration
        """
        self.game_id = game_id or str(uuid.uuid4())
        self.start_time = datetime.utcnow()

        self.log = {
            "schema_version": SCHEMA_VERSION,
            "game_id": self.game_id,
            "game_type": game_type,
            "created_at": self.start_time.isoformat() + "Z",
            "players": players or [],
            "config": config or {},
            "initial_state": None,
            "turns": [],
            "result": None,
            "duration_seconds": None,
        }

    def set_initial_state(self, state: Dict[str, Any]):
        """Set the initial game state (board layout, etc.)."""
        self.log["initial_state"] = state

    def record_turn(
        self,
        turn_number: int,
        player_id: str,
        phase: str,
        public_state: Dict[str, Any],
        private_states: Dict[str, Dict[str, Any]],
        valid_actions: List[Dict[str, Any]],
        action: Dict[str, Any],
        llm_decision: Optional[Dict[str, Any]] = None,
        action_result: Optional[Dict[str, Any]] = None,
    ):
        """
        Record a single turn.

        Args:
            turn_number: Turn index
            player_id: ID of player who acted
            phase: Game phase (e.g., "PLAY_TURN", "MOVE_ROBBER")
            public_state: Publicly visible state
            private_states: Private state for each player
            valid_actions: List of valid actions for this turn
            action: Action that was taken
            llm_decision: LLM decision record (if LLM player)
            action_result: Result of the action (dice rolls, etc.)
        """
        turn = {
            "turn_number": turn_number,
            "player_id": player_id,
            "phase": phase,
            "public_state": public_state,
            "private_states": private_states,
            "valid_actions": valid_actions,
            "llm_decision": llm_decision,
            "action": action,
            "action_result": action_result,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        self.log["turns"].append(turn)

    def finalize(
        self,
        termination_reason: str,
        winner_id: Optional[str] = None,
        final_scores: Optional[Dict[str, int]] = None,
    ):
        """
        Finalize the game log with results.

        Args:
            termination_reason: Why game ended ("victory", "turn_limit", "error")
            winner_id: ID of winning player (if any)
            final_scores: Final VP scores for each player
        """
        end_time = datetime.utcnow()
        duration = (end_time - self.start_time).total_seconds()

        # Calculate statistics
        stats = self._calculate_statistics()

        self.log["result"] = {
            "termination_reason": termination_reason,
            "winner_id": winner_id,
            "final_scores": final_scores or {},
            "total_turns": len(self.log["turns"]),
            "statistics": stats,
        }
        self.log["duration_seconds"] = duration

    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate game statistics from turn records."""
        stats = {
            "total_llm_calls": 0,
            "total_tokens_used": 0,
            "average_decision_time_ms": 0,
            "invalid_action_attempts": {},
            "actions_by_type": {},
            "actions_by_player": {},
        }

        decision_times = []

        for turn in self.log["turns"]:
            player_id = turn.get("player_id", "unknown")
            action = turn.get("action", {})
            action_type = action.get("action_type", "unknown")

            # Count actions by type
            stats["actions_by_type"][action_type] = stats["actions_by_type"].get(action_type, 0) + 1

            # Count actions by player
            stats["actions_by_player"][player_id] = stats["actions_by_player"].get(player_id, 0) + 1

            # LLM statistics
            llm_decision = turn.get("llm_decision")
            if llm_decision:
                stats["total_llm_calls"] += 1
                stats["total_tokens_used"] += llm_decision.get("total_tokens", 0)

                latency = llm_decision.get("latency_ms", 0)
                if latency > 0:
                    decision_times.append(latency)

                # Track parse failures
                if not llm_decision.get("parsed_successfully", True):
                    stats["invalid_action_attempts"][player_id] = (
                        stats["invalid_action_attempts"].get(player_id, 0) + 1
                    )

        if decision_times:
            stats["average_decision_time_ms"] = sum(decision_times) / len(decision_times)

        return stats

    def save(self, filepath: str, compress: bool = False):
        """
        Save the game log to a file.

        Args:
            filepath: Output file path
            compress: If True, compress with gzip
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        content = json.dumps(self.log, indent=2, default=str)

        if compress or filepath.endswith(".gz"):
            with gzip.open(path, "wt", encoding="utf-8") as f:
                f.write(content)
        else:
            path.write_text(content, encoding="utf-8")

    def to_dict(self) -> Dict[str, Any]:
        """Return the log as a dictionary."""
        return self.log.copy()


class GameLogReader:
    """
    Reader for game log files.

    Loads and provides access to game log data for replay and analysis.

    Example:
        reader = GameLogReader.load("./logs/game_abc123.json")

        # Iterate through turns
        for turn in reader.iter_turns():
            print(f"Turn {turn['turn_number']}: {turn['action']}")

        # Get specific turn
        turn_15 = reader.get_turn(15)

        # Get player's view at turn
        view = reader.get_player_view("player_0", 15)
    """

    def __init__(self, log_data: Dict[str, Any]):
        """
        Initialize reader with log data.

        Args:
            log_data: Parsed game log dictionary
        """
        self.log = log_data
        self._validate()

    @classmethod
    def load(cls, filepath: str) -> "GameLogReader":
        """
        Load a game log from file.

        Args:
            filepath: Path to log file (.json or .json.gz)

        Returns:
            GameLogReader instance
        """
        path = Path(filepath)

        if filepath.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(path.read_text(encoding="utf-8"))

        return cls(data)

    def _validate(self):
        """Validate log structure."""
        required_fields = ["schema_version", "game_id", "game_type", "turns"]
        for field in required_fields:
            if field not in self.log:
                raise ValueError(f"Invalid game log: missing field '{field}'")

    @property
    def game_id(self) -> str:
        return self.log["game_id"]

    @property
    def game_type(self) -> str:
        return self.log["game_type"]

    @property
    def players(self) -> List[Dict[str, Any]]:
        return self.log.get("players", [])

    @property
    def config(self) -> Dict[str, Any]:
        return self.log.get("config", {})

    @property
    def result(self) -> Optional[Dict[str, Any]]:
        return self.log.get("result")

    @property
    def total_turns(self) -> int:
        return len(self.log["turns"])

    def iter_turns(self):
        """Iterate through all turns."""
        yield from self.log["turns"]

    def get_turn(self, turn_number: int) -> Optional[Dict[str, Any]]:
        """Get a specific turn by number."""
        for turn in self.log["turns"]:
            if turn["turn_number"] == turn_number:
                return turn
        return None

    def get_turns_for_player(self, player_id: str) -> List[Dict[str, Any]]:
        """Get all turns for a specific player."""
        return [
            turn for turn in self.log["turns"]
            if turn["player_id"] == player_id
        ]

    def get_player_view(
        self,
        player_id: str,
        turn_number: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Get the game state from a player's perspective at a specific turn.

        This simulates "fog of war" - showing only what that player
        could see at that moment.

        Args:
            player_id: Player to get view for
            turn_number: Turn to get view at

        Returns:
            Dict with public_state and player's private_state, or None
        """
        turn = self.get_turn(turn_number)
        if turn is None:
            return None

        return {
            "public_state": turn["public_state"],
            "private_state": turn["private_states"].get(player_id, {}),
            "is_current_player": turn["player_id"] == player_id,
            "valid_actions": turn["valid_actions"] if turn["player_id"] == player_id else [],
        }

    def get_llm_decisions(self) -> List[Dict[str, Any]]:
        """Get all LLM decisions from the game."""
        decisions = []
        for turn in self.log["turns"]:
            if turn.get("llm_decision"):
                decisions.append({
                    "turn_number": turn["turn_number"],
                    "player_id": turn["player_id"],
                    **turn["llm_decision"],
                })
        return decisions

    def get_statistics(self) -> Dict[str, Any]:
        """Get game statistics."""
        if self.result:
            return self.result.get("statistics", {})
        return {}

    def to_dict(self) -> Dict[str, Any]:
        """Return the full log as a dictionary."""
        return self.log.copy()
