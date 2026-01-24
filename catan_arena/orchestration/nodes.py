"""
LangGraph node implementations for game execution.

Each node performs a specific step in the game loop:
1. get_board_state - Extract current state from environment
2. llm_decide - Call LLM to make a decision
3. parse_action - Parse LLM output to game action
4. update_env - Execute action and advance game
5. finalize_game - Record final results
"""

import time
import random
from datetime import datetime
from typing import Dict, Any, Literal, Optional

from catan_arena.orchestration.state import ArenaGameState, AgentDecision, TurnRecord


def get_board_state(state: ArenaGameState) -> Dict[str, Any]:
    """
    Node: Get Board State

    Extracts the current game state from the PettingZoo environment
    and prepares observations for the current player.
    """
    env = state.get("_env")
    if env is None:
        return {"error": "Environment not initialized"}

    # Get current agent
    current_agent = env.agent_selection

    # Get observation for current agent
    observation = env.observe(current_agent)

    # Find player index
    player_index = 0
    for i, player in enumerate(state["players"]):
        if player["id"] == current_agent:
            player_index = i
            break

    # Extract public and private states
    public_state = observation.get("public", {})
    private_state = observation.get("private", {})

    # Build private states dict (only current player has private info)
    private_states = {current_agent: private_state}

    # Get valid actions
    valid_actions = []
    action_mask = observation.get("action_mask", [])
    for i, valid in enumerate(action_mask):
        if valid:
            valid_actions.append({"action_index": i})

    # Format valid actions for prompt
    from catan_arena.adapters.catan_state_adapter import CatanStateAdapter
    adapter = CatanStateAdapter()

    # Get playable actions from env
    playable_actions = env.get_playable_actions()
    formatted_actions = adapter.format_valid_actions_for_prompt(playable_actions)

    return {
        "current_player_index": player_index,
        "current_public_state": public_state,
        "current_private_states": private_states,
        "current_valid_actions": formatted_actions,
        "phase": "main" if not public_state.get("is_initial_build_phase") else "setup",
    }


def llm_decide(state: ArenaGameState) -> Dict[str, Any]:
    """
    Node: LLM Decide

    Invokes the appropriate LLM for the current player to make a decision.
    """
    # Get current player
    player_index = state["current_player_index"]
    player = state["players"][player_index]
    player_type = player.get("type", "random")

    # If not an LLM player, skip to random/rule-based decision
    if player_type != "llm":
        return _make_non_llm_decision(state, player)

    # Get LLM player
    llm_player = state.get("_llm_players", {}).get(player["id"])
    if llm_player is None:
        return _make_non_llm_decision(state, player)

    # Build prompt and call LLM
    from catan_arena.adapters.catan_state_adapter import CatanStateAdapter

    adapter = CatanStateAdapter()
    prompt = adapter.state_to_prompt(
        public_state=state["current_public_state"],
        private_state=state["current_private_states"].get(player["id"], {}),
        valid_actions=state["current_valid_actions"],
        turn_history=state["turn_history"][-10:],
    )

    # Call LLM
    start_time = time.time()
    try:
        llm = llm_player.llm
        response = llm.invoke([
            {"role": "system", "content": adapter.format_system_prompt()},
            {"role": "user", "content": prompt},
        ])
        raw_output = response.content
        latency_ms = int((time.time() - start_time) * 1000)

        # Extract token usage
        token_usage = getattr(response, "usage_metadata", {}) or {}
        prompt_tokens = token_usage.get("input_tokens", 0)
        completion_tokens = token_usage.get("output_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens

        # Extract reasoning
        from catan_arena.adapters.catan_action_parser import CatanActionParser
        parser = CatanActionParser()
        reasoning = parser.extract_reasoning(raw_output)

        decision = AgentDecision(
            agent_id=player["id"],
            model_name=player.get("model", "unknown"),
            reasoning=reasoning,
            raw_output=raw_output,
            parsed_action={},
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            parsed_successfully=False,
            parse_error=None,
        )

        return {"pending_decision": decision}

    except Exception as e:
        # LLM call failed
        return {
            "pending_decision": AgentDecision(
                agent_id=player["id"],
                model_name=player.get("model", "unknown"),
                reasoning="",
                raw_output=f"LLM Error: {str(e)}",
                parsed_action={},
                latency_ms=int((time.time() - start_time) * 1000),
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                parsed_successfully=False,
                parse_error=str(e),
            )
        }


def _make_non_llm_decision(state: ArenaGameState, player: Dict) -> Dict[str, Any]:
    """Make a decision for non-LLM players (random, etc.)."""
    valid_actions = state["current_valid_actions"]

    if not valid_actions:
        return {"error": "No valid actions available"}

    # Random selection
    action = random.choice(valid_actions)

    return {
        "pending_decision": AgentDecision(
            agent_id=player["id"],
            model_name=player.get("type", "random"),
            reasoning="Random selection",
            raw_output="",
            parsed_action=action,
            latency_ms=0,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            parsed_successfully=True,
            parse_error=None,
        )
    }


def parse_action(state: ArenaGameState) -> Dict[str, Any]:
    """
    Node: Parse Action

    Parses the LLM's raw output into a valid game action.
    """
    decision = state.get("pending_decision")
    if decision is None:
        return {"error": "No pending decision to parse"}

    # If already parsed (non-LLM player), skip
    if decision.get("parsed_successfully"):
        return {"pending_action": decision["parsed_action"]}

    # Parse LLM output
    from catan_arena.adapters.catan_action_parser import CatanActionParser

    parser = CatanActionParser(fallback_to_random=True)
    valid_actions = state["current_valid_actions"]

    try:
        parsed = parser.parse(decision["raw_output"], valid_actions)
        decision["parsed_action"] = parsed
        decision["parsed_successfully"] = True

        return {
            "pending_decision": decision,
            "pending_action": parsed,
        }

    except Exception as e:
        # Parse failed - use random fallback
        fallback = random.choice(valid_actions) if valid_actions else {"action_type": "END_TURN"}
        decision["parsed_action"] = fallback
        decision["parsed_successfully"] = False
        decision["parse_error"] = str(e)

        return {
            "pending_decision": decision,
            "pending_action": fallback,
        }


def update_env(state: ArenaGameState) -> Dict[str, Any]:
    """
    Node: Update Environment

    Applies the parsed action to the PettingZoo environment
    and creates a turn record.
    """
    env = state.get("_env")
    if env is None:
        return {"error": "Environment not initialized"}

    action = state.get("pending_action")
    if action is None:
        return {"error": "No pending action to execute"}

    # Convert action to environment format
    action_type = action.get("action_type")
    action_value = action.get("value")

    # Find action index
    from catan_arena.envs.catan_pettingzoo import ACTIONS_ARRAY
    from catanatron.models.enums import ActionType

    action_int = None
    try:
        action_type_enum = ActionType[action_type]

        # Normalize value for matching
        if isinstance(action_value, list):
            action_value = tuple(sorted(action_value)) if action_type == "BUILD_ROAD" else tuple(action_value)

        for i, (at, av) in enumerate(ACTIONS_ARRAY):
            if at == action_type_enum:
                if av == action_value or (av is None and action_value is None):
                    action_int = i
                    break
    except Exception:
        pass

    # Execute action
    action_result = {}
    if action_int is not None:
        env.step(action_int)
    else:
        # Invalid action - step with None (will trigger fallback)
        env.step(None)
        action_result["invalid"] = True

    # Create turn record
    player = state["players"][state["current_player_index"]]
    turn_record = TurnRecord(
        turn_number=state["turn_number"],
        player_id=player["id"],
        phase=state["phase"],
        public_state=state["current_public_state"],
        private_states=state["current_private_states"],
        valid_actions=state["current_valid_actions"],
        decision=state.get("pending_decision"),
        action=action,
        action_result=action_result,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )

    return {
        "turn_number": state["turn_number"] + 1,
        "turn_history": [turn_record],
        "pending_decision": None,
        "pending_action": None,
        "current_public_state": None,
        "current_private_states": None,
        "current_valid_actions": None,
    }


def check_game_over(state: ArenaGameState) -> Literal["continue", "finished"]:
    """
    Conditional Edge: Check if game should continue or end.
    """
    env = state.get("_env")
    if env is None:
        return "finished"

    # Check if any agent is done
    for agent in env.agents:
        if env.terminations.get(agent) or env.truncations.get(agent):
            return "finished"

    # Check turn limit
    max_turns = state.get("_max_turns", 1000)
    if state["turn_number"] >= max_turns:
        return "finished"

    return "continue"


def finalize_game(state: ArenaGameState) -> Dict[str, Any]:
    """
    Node: Finalize Game

    Records final scores and determines winner.
    """
    env = state.get("_env")
    if env is None:
        return {
            "phase": "finished",
            "termination_reason": "error",
            "error": "Environment not available",
        }

    # Get final scores
    final_scores = {}
    winner = None
    max_score = -1

    for agent in env.possible_agents:
        reward = env.rewards.get(agent, 0)
        if reward > max_score:
            max_score = reward
            winner = agent

        # Get actual VP from game
        color = env.agent_colors.get(agent)
        if color and env.game:
            from catanatron.state_functions import get_actual_victory_points
            vp = get_actual_victory_points(env.game.state, color)
            final_scores[agent] = vp

    # Determine termination reason
    termination_reason = "turn_limit"
    for agent in env.agents:
        if env.terminations.get(agent):
            termination_reason = "victory"
            break

    return {
        "phase": "finished",
        "winner": winner if termination_reason == "victory" else None,
        "final_scores": final_scores,
        "termination_reason": termination_reason,
    }
