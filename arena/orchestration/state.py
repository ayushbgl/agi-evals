"""
State definitions for game-agnostic LangGraph orchestration.

Defines the shared state schema used across all nodes in the
game execution graph, supporting multiple game types.
"""

from typing import TypedDict, Literal, List, Optional, Dict, Any, Annotated
from operator import add


class AgentDecision(TypedDict):
    """Record of a single agent's decision."""

    agent_id: str
    model_name: str
    reasoning: str  # Chain-of-thought
    raw_output: str
    parsed_action: Dict[str, Any]
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    parsed_successfully: bool
    parse_error: Optional[str]


class TurnRecord(TypedDict):
    """Complete record of one game turn."""

    turn_number: int
    player_id: str
    role: str  # Player's role (e.g., "spymaster", "operative", "player")
    phase: str
    public_state: Dict[str, Any]
    private_states: Dict[str, Dict[str, Any]]
    valid_actions: List[Dict[str, Any]]
    decision: Optional[AgentDecision]
    action: Dict[str, Any]
    action_result: Optional[Dict[str, Any]]
    timestamp: str


class ArenaGameState(TypedDict):
    """
    LangGraph shared state for game execution.

    This state is passed through all nodes and accumulates
    the complete game history. Designed to be game-agnostic.
    """

    # Game identification
    game_id: str
    game_type: str  # "catan", "codenames", etc.

    # Player configuration
    players: List[Dict[str, Any]]  # [{id, model, team, role, type}, ...]
    current_player_index: int

    # Game state
    turn_number: int
    phase: str  # Game-specific phase

    # Current turn working state (cleared each turn)
    current_player_id: Optional[str]
    current_role: Optional[str]
    current_public_state: Optional[Dict[str, Any]]
    current_private_states: Optional[Dict[str, Dict[str, Any]]]
    current_valid_actions: Optional[List[Dict[str, Any]]]
    pending_decision: Optional[AgentDecision]
    pending_action: Optional[Dict[str, Any]]

    # Accumulating history (Annotated for append-only updates)
    turn_history: Annotated[List[TurnRecord], add]

    # Terminal state
    winner: Optional[str]
    final_scores: Optional[Dict[str, Any]]
    termination_reason: Optional[str]

    # Error tracking
    error: Optional[str]

    # Internal references (not serialized)
    # _game: Game instance
    # _state_adapter: StateAdapter instance
    # _action_parser: ActionParser instance
    # _llm_players: Dict of LLM players
    # _agents: Dict of all agents


def create_initial_state(
    game_id: str,
    players: List[Dict[str, Any]],
    game_type: str,
) -> ArenaGameState:
    """
    Create the initial state for a new game.

    Args:
        game_id: Unique game identifier
        players: List of player configurations
        game_type: Type of game ("catan", "codenames", etc.)

    Returns:
        Initial ArenaGameState
    """
    return ArenaGameState(
        game_id=game_id,
        game_type=game_type,
        players=players,
        current_player_index=0,
        turn_number=0,
        phase="setup",
        current_player_id=None,
        current_role=None,
        current_public_state=None,
        current_private_states=None,
        current_valid_actions=None,
        pending_decision=None,
        pending_action=None,
        turn_history=[],
        winner=None,
        final_scores=None,
        termination_reason=None,
        error=None,
    )
