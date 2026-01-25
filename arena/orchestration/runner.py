"""
Game-agnostic runner for Arena games.

Provides the main entry point for running LLM vs LLM (or other agent types)
games with full orchestration and logging.
"""

import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from arena.config import ArenaConfig, PlayerConfig, create_config
from arena.registry import (
    get_game_class,
    get_state_adapter_class,
    get_action_parser_class,
)
from arena.orchestration.state import create_initial_state, ArenaGameState, TurnRecord, AgentDecision
from arena.llm.manual_agent import ManualAgent
from core.agent import RandomAgent


def run_arena_game(
    config: ArenaConfig,
    save_log: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run a complete Arena game.

    This is a simplified synchronous runner that doesn't use LangGraph,
    suitable for basic game execution and testing.

    Args:
        config: Arena configuration with player settings
        save_log: Whether to save the game log
        verbose: Whether to print progress

    Returns:
        Dict containing game results and statistics
    """
    game_id = str(uuid.uuid4())

    if verbose:
        print(f"Starting {config.game_type} game {game_id}")
        print(f"Players: {[p.id for p in config.players]}")

    # Initialize game based on game_type
    game = _create_game(config)

    # Initialize adapters
    StateAdapterClass = get_state_adapter_class(config.game_type)
    ActionParserClass = get_action_parser_class(config.game_type)

    state_adapter = StateAdapterClass()
    action_parser = ActionParserClass()

    # Initialize agents for each player
    agents = _create_agents(config)

    # Build player list for state
    players = [
        {
            "id": p.id,
            "type": p.type,
            "model": p.model,
            "team": p.team,
            "role": p.role,
        }
        for p in config.players
    ]

    # Create initial state
    state = create_initial_state(game_id, players, config.game_type)

    # Track turn history
    turn_history = []
    turn_number = 0

    # Main game loop
    while not game.is_over() and turn_number < config.max_turns:
        turn_number += 1

        # Get current player
        current_player_id = game.get_current_player()
        if not current_player_id:
            break

        current_role = game.get_current_role()

        # Get game state
        public_state = game.get_public_state()
        private_state = game.get_private_state(current_player_id)
        valid_actions = game.get_available_actions()

        if verbose:
            print(f"\nTurn {turn_number}: {current_player_id} ({current_role})")

        # Get agent for current player
        agent = agents.get(current_player_id)
        if agent is None:
            if verbose:
                print(f"  No agent for {current_player_id}, using random")
            agent = RandomAgent(current_player_id)

        # Generate prompt
        prompt = state_adapter.state_to_prompt(
            public_state=public_state,
            private_state=private_state,
            valid_actions=valid_actions,
            turn_history=turn_history[-10:],
        )

        system_prompt = state_adapter.format_system_prompt(
            role=current_role,
            team=private_state.get("team"),
            clue=public_state.get("current_clue", {}).get("word") if public_state.get("current_clue") else None,
            number=public_state.get("current_clue", {}).get("number") if public_state.get("current_clue") else None,
        )

        # Get decision from agent
        start_time = datetime.now()
        try:
            decision_result = agent.decide(
                prompt=prompt,
                valid_actions=valid_actions,
                system_prompt=system_prompt,
                game_type=config.game_type,
                role=current_role,
                turn_number=turn_number,
            )
            latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            action = decision_result.get("action", {})
            reasoning = decision_result.get("reasoning", "")
            raw_output = decision_result.get("raw_output", "")

            # Parse/validate action if needed
            try:
                if hasattr(action_parser, "parse") and isinstance(raw_output, str) and raw_output:
                    parsed_action = action_parser.parse(raw_output, valid_actions)
                else:
                    parsed_action = action
                parse_success = True
                parse_error = None
            except Exception as e:
                parsed_action = action
                parse_success = False
                parse_error = str(e)

            decision = AgentDecision(
                agent_id=current_player_id,
                model_name=str(agent.agent_type.value),
                reasoning=reasoning,
                raw_output=raw_output,
                parsed_action=parsed_action,
                latency_ms=latency_ms,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                parsed_successfully=parse_success,
                parse_error=parse_error,
            )

        except Exception as e:
            if verbose:
                print(f"  Agent error: {e}")
            # Use random action as fallback
            import random
            action = random.choice(valid_actions) if valid_actions else {}
            parsed_action = action
            decision = AgentDecision(
                agent_id=current_player_id,
                model_name="error_fallback",
                reasoning=f"Error: {e}",
                raw_output="",
                parsed_action=action,
                latency_ms=0,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                parsed_successfully=False,
                parse_error=str(e),
            )

        # Execute action
        action_result, game_over = game.step(parsed_action)

        if verbose:
            action_type = parsed_action.get("action_type", "?")
            print(f"  Action: {action_type}")
            if game_over:
                print(f"  Game Over!")

        # Record turn
        turn_record = TurnRecord(
            turn_number=turn_number,
            player_id=current_player_id,
            role=current_role,
            phase=public_state.get("current_phase", "main"),
            public_state=public_state,
            private_states={current_player_id: private_state},
            valid_actions=valid_actions,
            decision=decision,
            action=parsed_action,
            action_result=action_result,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )
        turn_history.append(turn_record)

    # Get final results
    winner = game.get_winner()
    scores = game.get_scores()

    # Determine termination reason
    if game.is_over():
        termination_reason = "victory" if winner else "draw"
    else:
        termination_reason = "turn_limit"

    # Build result
    result = {
        "game_id": game_id,
        "game_type": config.game_type,
        "winner": winner,
        "final_scores": scores,
        "total_turns": turn_number,
        "termination_reason": termination_reason,
        "turn_history": turn_history,
    }

    # Save log if requested
    if save_log:
        log_dir = Path(config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{config.game_type}_{game_id}.json"

        import json
        log_data = {
            "game_id": game_id,
            "game_type": config.game_type,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "players": players,
            "config": config.model_dump() if hasattr(config, "model_dump") else {},
            "result": {
                "winner": winner,
                "final_scores": scores,
                "total_turns": turn_number,
                "termination_reason": termination_reason,
            },
            "turns": [
                {
                    "turn_number": t["turn_number"],
                    "player_id": t["player_id"],
                    "role": t["role"],
                    "action": t["action"],
                    "action_result": t["action_result"],
                }
                for t in turn_history
            ],
        }

        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2, default=str)

        result["log_path"] = str(log_path)

        if verbose:
            print(f"\nGame log saved to {log_path}")

    if verbose:
        print(f"\nGame finished: {termination_reason}")
        print(f"Winner: {winner}")
        print(f"Scores: {scores}")

    return result


def run_simple_game(
    game_type: str,
    player_configs: List[Dict[str, Any]],
    max_turns: int = 500,
    seed: Optional[int] = None,
    log_dir: str = "./game_logs",
    verbose: bool = True,
    **game_kwargs,
) -> Dict[str, Any]:
    """
    Simplified interface for running a game.

    Args:
        game_type: Type of game ("catan", "codenames", etc.)
        player_configs: List of player config dicts
        max_turns: Maximum turns before truncation
        seed: Random seed
        log_dir: Directory for game logs
        verbose: Print progress
        **game_kwargs: Additional game-specific config

    Returns:
        Game result dict
    """
    config = create_config(
        game_type=game_type,
        player_configs=player_configs,
        max_turns=max_turns,
        seed=seed,
        log_dir=log_dir,
        verbose=verbose,
        game_config=game_kwargs if game_kwargs else None,
    )

    return run_arena_game(config, verbose=verbose)


def _create_game(config: ArenaConfig):
    """Create a game instance based on config."""
    game_type = config.game_type

    if game_type == "codenames":
        from games.codenames.game import CodenamesGame

        # Extract player assignments
        red_spymaster = None
        red_operatives = []
        blue_spymaster = None
        blue_operatives = []

        for p in config.players:
            if p.team == "red":
                if p.role == "spymaster":
                    red_spymaster = p.id
                else:
                    red_operatives.append(p.id)
            elif p.team == "blue":
                if p.role == "spymaster":
                    blue_spymaster = p.id
                else:
                    blue_operatives.append(p.id)

        game = CodenamesGame(
            red_spymaster=red_spymaster or "red_spymaster",
            red_operatives=red_operatives or ["red_operative"],
            blue_spymaster=blue_spymaster or "blue_spymaster",
            blue_operatives=blue_operatives or ["blue_operative"],
            seed=config.seed,
        )
        return game

    elif game_type == "catan":
        # For Catan, use the existing PettingZoo environment
        from catan_arena.envs.catan_pettingzoo import CatanAECEnv

        game_config = config.game_config or {}
        env = CatanAECEnv(
            num_players=len(config.players),
            map_type=game_config.get("map_type", "BASE"),
            vps_to_win=game_config.get("vps_to_win", 10),
            max_turns=config.max_turns,
        )
        env.reset(seed=config.seed)
        return env

    else:
        raise ValueError(f"Unknown game type: {game_type}")


def _create_agents(config: ArenaConfig) -> Dict[str, Any]:
    """Create agents for all players."""
    agents = {}

    for player_config in config.players:
        if player_config.type == "manual":
            agents[player_config.id] = ManualAgent(
                agent_id=player_config.id,
            )
        elif player_config.type == "random":
            agents[player_config.id] = RandomAgent(
                agent_id=player_config.id,
                seed=config.seed,
            )
        elif player_config.type == "llm":
            # LLM agents would be created here
            # For now, fall back to random
            agents[player_config.id] = RandomAgent(
                agent_id=player_config.id,
                seed=config.seed,
            )
        # Add more agent types as needed

    return agents
