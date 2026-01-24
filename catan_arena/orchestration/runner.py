"""
Game execution runner for Catan-Arena.

Provides the main entrypoint for running LLM vs LLM games
with full orchestration, logging, and optional checkpointing.
"""

import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path

from catan_arena.config import ArenaConfig, PlayerConfig
from catan_arena.envs.catan_pettingzoo import CatanAECEnv
from catan_arena.orchestration.state import ArenaGameState, create_initial_state
from catan_arena.orchestration.graph import build_arena_graph, create_checkpointer
from catan_arena.storage.game_log import GameLogWriter
from catan_arena.llm.player import LLMPlayer


def run_arena_game(
    config: ArenaConfig,
    save_log: bool = True,
    use_checkpoints: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run a complete Catan-Arena game.

    Args:
        config: Arena configuration with player settings
        save_log: Whether to save the game log
        use_checkpoints: Whether to enable state checkpointing
        verbose: Whether to print progress

    Returns:
        Dict containing game results and statistics
    """
    game_id = str(uuid.uuid4())

    if verbose:
        print(f"Starting game {game_id}")
        print(f"Players: {[p.id for p in config.players]}")

    # Initialize environment
    env = CatanAECEnv(
        num_players=len(config.players),
        map_type=config.game_config.map_type,
        vps_to_win=config.game_config.vps_to_win,
        max_turns=config.max_turns,
    )
    env.reset(seed=config.seed)

    # Initialize LLM players
    llm_players = {}
    for player_config in config.players:
        if player_config.type == "llm" and player_config.model:
            llm_config = player_config.llm_config.model_dump() if player_config.llm_config else {}
            llm_player = LLMPlayer(
                color=env.agent_colors[player_config.id],
                model_name=player_config.model,
                llm_config=llm_config,
            )
            llm_players[player_config.id] = llm_player

    # Build player list for state
    players = [
        {
            "id": p.id,
            "color": p.color,
            "type": p.type,
            "model": p.model,
        }
        for p in config.players
    ]

    # Initialize game log
    log_writer = GameLogWriter(
        game_id=game_id,
        game_type="catan",
        players=players,
        config={
            "map_type": config.game_config.map_type,
            "vps_to_win": config.game_config.vps_to_win,
            "max_turns": config.max_turns,
            "seed": config.seed,
        }
    )

    # Set initial state
    initial_board = env._serialize_board()
    log_writer.set_initial_state(initial_board)

    # Build orchestration graph
    checkpointer = create_checkpointer(config.checkpoint_dir) if use_checkpoints else None
    graph = build_arena_graph(checkpointer)

    # Create initial state
    state = create_initial_state(game_id, players)
    state["_env"] = env  # Store env reference (not serialized)
    state["_llm_players"] = llm_players
    state["_max_turns"] = config.max_turns

    # Run game through LangGraph
    graph_config = {"configurable": {"thread_id": game_id}} if use_checkpoints else {}

    try:
        final_state = graph.invoke(state, graph_config)
    except Exception as e:
        if verbose:
            print(f"Game error: {e}")
        final_state = state
        final_state["error"] = str(e)
        final_state["phase"] = "finished"
        final_state["termination_reason"] = "error"

    # Record turns to log
    for turn in final_state.get("turn_history", []):
        log_writer.record_turn(
            turn_number=turn["turn_number"],
            player_id=turn["player_id"],
            phase=turn["phase"],
            public_state=turn["public_state"],
            private_states=turn["private_states"],
            valid_actions=turn["valid_actions"],
            action=turn["action"],
            llm_decision=turn.get("decision"),
            action_result=turn.get("action_result"),
        )

    # Finalize log
    log_writer.finalize(
        termination_reason=final_state.get("termination_reason", "unknown"),
        winner_id=final_state.get("winner"),
        final_scores=final_state.get("final_scores"),
    )

    # Save log
    if save_log:
        log_dir = Path(config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{game_id}.json"
        log_writer.save(str(log_path))

        if verbose:
            print(f"Game log saved to {log_path}")

    # Build result
    result = {
        "game_id": game_id,
        "winner": final_state.get("winner"),
        "final_scores": final_state.get("final_scores", {}),
        "total_turns": final_state.get("turn_number", 0),
        "termination_reason": final_state.get("termination_reason"),
        "log_path": str(log_path) if save_log else None,
        "statistics": log_writer.to_dict().get("result", {}).get("statistics", {}),
    }

    if verbose:
        print(f"Game finished: {result['termination_reason']}")
        print(f"Winner: {result['winner']}")
        print(f"Scores: {result['final_scores']}")

    return result


def run_simple_game(
    player_configs: List[Dict[str, Any]],
    map_type: str = "BASE",
    vps_to_win: int = 10,
    max_turns: int = 500,
    seed: Optional[int] = None,
    log_dir: str = "./game_logs",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Simplified interface for running a game.

    Args:
        player_configs: List of player config dicts, e.g.:
            [
                {"id": "player_0", "color": "RED", "type": "llm", "model": "claude-3-opus-20240229"},
                {"id": "player_1", "color": "BLUE", "type": "random"},
            ]
        map_type: Map template ("BASE" or "MINI")
        vps_to_win: Victory points to win
        max_turns: Maximum turns before truncation
        seed: Random seed
        log_dir: Directory for game logs
        verbose: Print progress

    Returns:
        Game result dict
    """
    from catan_arena.config import PlayerConfig, CatanGameConfig, ArenaConfig, LLMConfig

    # Convert player configs
    players = []
    for pc in player_configs:
        llm_config = None
        if pc.get("llm_config"):
            llm_config = LLMConfig(**pc["llm_config"])

        players.append(PlayerConfig(
            id=pc["id"],
            color=pc["color"],
            type=pc["type"],
            model=pc.get("model"),
            llm_config=llm_config,
        ))

    config = ArenaConfig(
        players=players,
        game_config=CatanGameConfig(
            map_type=map_type,
            vps_to_win=vps_to_win,
        ),
        max_turns=max_turns,
        seed=seed,
        log_dir=log_dir,
        verbose=verbose,
    )

    return run_arena_game(config, verbose=verbose)


def run_tournament(
    player_configs: List[Dict[str, Any]],
    num_games: int = 10,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run a tournament of multiple games.

    Args:
        player_configs: List of player configurations
        num_games: Number of games to run
        **kwargs: Additional arguments passed to run_simple_game

    Returns:
        Tournament results with win counts and statistics
    """
    results = []
    wins = {pc["id"]: 0 for pc in player_configs}

    for i in range(num_games):
        print(f"\n=== Game {i+1}/{num_games} ===")

        result = run_simple_game(player_configs, **kwargs)
        results.append(result)

        if result.get("winner"):
            wins[result["winner"]] = wins.get(result["winner"], 0) + 1

    # Calculate statistics
    total_turns = sum(r.get("total_turns", 0) for r in results)
    avg_turns = total_turns / len(results) if results else 0

    return {
        "num_games": num_games,
        "wins": wins,
        "avg_turns": avg_turns,
        "results": results,
    }
