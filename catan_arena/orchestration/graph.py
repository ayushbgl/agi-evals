"""
LangGraph workflow assembly for game execution.

Builds the cyclic graph that orchestrates the game loop:
get_board_state → llm_decide → parse_action → update_env → [continue/finished]
"""

from typing import Optional

from langgraph.graph import StateGraph, END

from catan_arena.orchestration.state import ArenaGameState
from catan_arena.orchestration.nodes import (
    get_board_state,
    llm_decide,
    parse_action,
    update_env,
    check_game_over,
    finalize_game,
)


def build_arena_graph(checkpointer=None) -> StateGraph:
    """
    Build the LangGraph workflow for game execution.

    The graph implements a cyclic game loop where:
    1. get_board_state extracts current observation
    2. llm_decide calls the LLM for the current player
    3. parse_action parses LLM output to game action
    4. update_env executes the action
    5. check_game_over determines if game continues
    6. finalize_game records final results

    Args:
        checkpointer: Optional LangGraph checkpointer for state persistence

    Returns:
        Compiled StateGraph ready for execution
    """
    # Create workflow
    workflow = StateGraph(ArenaGameState)

    # Add nodes
    workflow.add_node("get_board_state", get_board_state)
    workflow.add_node("llm_decide", llm_decide)
    workflow.add_node("parse_action", parse_action)
    workflow.add_node("update_env", update_env)
    workflow.add_node("finalize_game", finalize_game)

    # Define edges (the game loop)
    workflow.set_entry_point("get_board_state")
    workflow.add_edge("get_board_state", "llm_decide")
    workflow.add_edge("llm_decide", "parse_action")
    workflow.add_edge("parse_action", "update_env")

    # Conditional edge: continue game or finish
    workflow.add_conditional_edges(
        "update_env",
        check_game_over,
        {
            "continue": "get_board_state",  # Loop back
            "finished": "finalize_game",    # Exit loop
        }
    )

    workflow.add_edge("finalize_game", END)

    # Compile with optional checkpointer
    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()


def create_checkpointer(checkpoint_dir: str = "./checkpoints"):
    """
    Create a checkpointer for state persistence.

    Args:
        checkpoint_dir: Directory for checkpoint storage

    Returns:
        LangGraph checkpointer instance
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        import os

        os.makedirs(checkpoint_dir, exist_ok=True)
        db_path = os.path.join(checkpoint_dir, "games.db")
        return SqliteSaver.from_conn_string(f"sqlite:///{db_path}")
    except ImportError:
        # Fall back to memory saver
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()
