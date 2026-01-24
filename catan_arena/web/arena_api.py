"""
Arena API - Flask endpoints for LLM-enabled Catan games.

Extends the base Catanatron web API with LLM player support.
"""

import json
import logging
from typing import Dict, Optional
from flask import Blueprint, Response, jsonify, abort, request

from catanatron.game import Game
from catanatron.models.player import Color, RandomPlayer
from catanatron.models.enums import ActionType
from catanatron.json import GameEncoder, action_from_json
from catanatron.players.minimax import AlphaBetaPlayer

from catan_arena.llm.player import LLMPlayer
from catan_arena.adapters.catan_state_adapter import CatanStateAdapter
from catan_arena.adapters.catan_action_parser import CatanActionParser
from catan_arena.adapters.board_renderer import render_board_compact, get_node_resource_info

# In-memory game storage (for simplicity - use Redis/DB in production)
ARENA_GAMES: Dict[str, Game] = {}
GAME_HISTORY: Dict[str, list] = {}  # game_id -> list of actions
LAST_DICE_ROLL: Dict[str, tuple] = {}  # game_id -> (dice1, dice2)

bp = Blueprint("arena_api", __name__, url_prefix="/api/arena")

state_adapter = CatanStateAdapter()
action_parser = CatanActionParser(fallback_to_random=False)


class WebLLMPlayer:
    """
    LLM player that generates prompts for web UI.

    Unlike the auto-playing LLMPlayer, this one:
    1. Generates a prompt when it's their turn
    2. Waits for response via API
    3. Validates and executes the action
    """

    def __init__(self, color: Color, name: str = "LLM"):
        self.color = color
        self.name = name
        self.is_bot = False  # Treated like human - waits for input

    def __repr__(self):
        return f"WebLLMPlayer({self.color}, {self.name})"


def arena_player_factory(player_key):
    """Create player from config. Supports: LLM, HUMAN, RANDOM, CATANATRON"""
    player_type, color = player_key

    if player_type.startswith("LLM"):
        # Format: "LLM:ModelName" or just "LLM"
        parts = player_type.split(":", 1)
        name = parts[1] if len(parts) > 1 else "LLM"
        return WebLLMPlayer(color, name)
    elif player_type == "RANDOM":
        return RandomPlayer(color)
    elif player_type == "CATANATRON":
        return AlphaBetaPlayer(color, 2, True)
    elif player_type == "HUMAN":
        # Human player - manual input via UI
        from catanatron.players.value import ValueFunctionPlayer
        return ValueFunctionPlayer(color, is_bot=False)
    else:
        raise ValueError(f"Invalid player type: {player_type}")


def get_arena_game(game_id: str) -> Optional[Game]:
    """Get game from storage."""
    return ARENA_GAMES.get(game_id)


def auto_roll_if_needed(game: Game) -> Optional[dict]:
    """
    Auto-execute ROLL action if that's the only option.
    Returns roll info if rolled, None otherwise.
    """
    playable_actions = game.playable_actions
    roll_actions = [a for a in playable_actions if a.action_type == ActionType.ROLL]

    if roll_actions:
        action = roll_actions[0]
        action_record = game.execute(action)

        # Capture dice result
        if action_record and action_record.result:
            dice = action_record.result
            LAST_DICE_ROLL[game.id] = dice

            # Record to history
            if game.id not in GAME_HISTORY:
                GAME_HISTORY[game.id] = []

            current_player = game.state.current_player()
            GAME_HISTORY[game.id].append({
                "player": current_player.color.name,
                "action_type": "ROLL",
                "value": list(dice),
                "total": sum(dice),
            })

            return {
                "dice": list(dice),
                "total": sum(dice),
                "player": current_player.color.name,
            }

    return None


def record_action(game: Game, action, player_color: str):
    """Record an action to game history."""
    if game.id not in GAME_HISTORY:
        GAME_HISTORY[game.id] = []

    GAME_HISTORY[game.id].append({
        "player": player_color,
        "action_type": action.action_type.name,
        "value": action.value,
    })


def save_arena_game(game: Game):
    """Save game to storage."""
    ARENA_GAMES[game.id] = game


def generate_llm_prompt(game: Game, player: WebLLMPlayer) -> dict:
    """Generate prompt data for an LLM player."""
    playable_actions = game.playable_actions

    # Extract state for prompt
    state = game.state
    board = state.board
    player_idx = next(i for i, p in enumerate(state.players) if p.color == player.color)

    public_state = {
        "turn_number": state.num_turns,
        "current_player": player.color.name,
        "phase": "INITIAL_BUILD" if state.is_initial_build_phase else "MAIN",
        "robber": list(board.robber_coordinate) if board.robber_coordinate else None,
        "players": {},
    }

    # Add dice roll info if available
    dice_roll = LAST_DICE_ROLL.get(game.id)
    if dice_roll and not state.is_initial_build_phase:
        public_state["last_dice_roll"] = {
            "dice": list(dice_roll),
            "total": sum(dice_roll),
        }

    # Add player summaries
    for i, p in enumerate(state.players):
        prefix = f"P{i}_"
        public_state["players"][p.color.name] = {
            "victory_points": state.player_state.get(f"{prefix}VICTORY_POINTS", 0),
            "resource_count": sum(
                state.player_state.get(f"{prefix}{r}_IN_HAND", 0)
                for r in ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
            ),
            "dev_cards": state.player_state.get(f"{prefix}NUM_DEVCARD", 0),
        }

    # Private state for this player
    prefix = f"P{player_idx}_"
    private_state = {
        "hand_resources": {
            "WOOD": state.player_state.get(f"{prefix}WOOD_IN_HAND", 0),
            "BRICK": state.player_state.get(f"{prefix}BRICK_IN_HAND", 0),
            "SHEEP": state.player_state.get(f"{prefix}SHEEP_IN_HAND", 0),
            "WHEAT": state.player_state.get(f"{prefix}WHEAT_IN_HAND", 0),
            "ORE": state.player_state.get(f"{prefix}ORE_IN_HAND", 0),
        },
        "hand_dev_cards": {
            "KNIGHT": state.player_state.get(f"{prefix}KNIGHT_IN_HAND", 0),
            "VICTORY_POINT": state.player_state.get(f"{prefix}VICTORY_POINT_IN_HAND", 0),
            "ROAD_BUILDING": state.player_state.get(f"{prefix}ROAD_BUILDING_IN_HAND", 0),
            "YEAR_OF_PLENTY": state.player_state.get(f"{prefix}YEAR_OF_PLENTY_IN_HAND", 0),
            "MONOPOLY": state.player_state.get(f"{prefix}MONOPOLY_IN_HAND", 0),
        },
    }

    # Add board information to public_state
    board_text = render_board_compact(game)
    node_info = get_node_resource_info(game)

    # Add board as structured data
    public_state["board_ascii"] = board_text
    public_state["node_resources"] = {
        str(k): v for k, v in node_info.items()
    }

    # Format valid actions (excluding ROLL since it's automatic)
    valid_actions = []
    for action in playable_actions:
        if action.action_type != ActionType.ROLL:
            valid_actions.append({
                "action_type": action.action_type.name,
                "value": action.value,
            })

    # Get game history
    game_history = GAME_HISTORY.get(game.id, [])

    # Generate full prompt with board
    prompt_text = state_adapter.state_to_prompt(
        public_state=public_state,
        private_state=private_state,
        valid_actions=valid_actions,
        turn_history=game_history[-15:],  # Last 15 actions
    )

    # Prepend board ASCII to prompt
    prompt_text = board_text + "\n\n" + prompt_text

    return {
        "player_name": player.name,
        "player_color": player.color.name,
        "prompt": prompt_text,
        "valid_actions": valid_actions,
        "action_types": list(set(a["action_type"] for a in valid_actions)),
    }


# ===== API Endpoints =====

@bp.route("/games", methods=["POST"])
def create_arena_game():
    """
    Create a new Arena game.

    Body: {"players": ["LLM:Claude", "LLM:GPT4", "RANDOM", "RANDOM"]}
    """
    if not request.is_json or "players" not in request.json:
        abort(400, description="Missing 'players' in request body")

    player_types = request.json["players"]
    if len(player_types) < 2 or len(player_types) > 4:
        abort(400, description="Must have 2-4 players")

    # Create players
    colors = list(Color)[:len(player_types)]
    players = [arena_player_factory((pt, c)) for pt, c in zip(player_types, colors)]

    # Create game
    game = Game(players=players)
    save_arena_game(game)

    # Initialize history
    GAME_HISTORY[game.id] = []
    LAST_DICE_ROLL[game.id] = None

    # Build player info
    player_info = []
    for p in players:
        info = {
            "color": p.color.name,
            "type": type(p).__name__,
        }
        if isinstance(p, WebLLMPlayer):
            info["name"] = p.name
            info["is_llm"] = True
        else:
            info["is_llm"] = False
        player_info.append(info)

    return jsonify({
        "game_id": game.id,
        "players": player_info,
    })


@bp.route("/games/<game_id>", methods=["GET"])
def get_arena_game_state(game_id: str):
    """Get current game state with LLM prompt if applicable."""
    game = get_arena_game(game_id)
    if not game:
        abort(404, description="Game not found")

    # Auto-roll if needed
    roll_info = auto_roll_if_needed(game)
    if roll_info:
        save_arena_game(game)

    current_player = game.state.current_player()

    response_data = {
        "game_id": game.id,
        "state": json.loads(json.dumps(game, cls=GameEncoder)),
        "current_player": current_player.color.name,
        "is_finished": game.winning_color() is not None,
        "winner": game.winning_color().name if game.winning_color() else None,
    }

    # Add roll info if just rolled
    if roll_info:
        response_data["dice_rolled"] = roll_info

    # If current player is LLM, include prompt
    if isinstance(current_player, WebLLMPlayer):
        response_data["llm_prompt"] = generate_llm_prompt(game, current_player)
        response_data["waiting_for_llm"] = True
    else:
        response_data["waiting_for_llm"] = False

    return jsonify(response_data)


@bp.route("/games/<game_id>/llm-action", methods=["POST"])
def submit_llm_action(game_id: str):
    """
    Submit an LLM's action.

    Body: {"response": "LLM's full response text"}
    or:   {"action": {"action_type": "BUILD_ROAD", "value": [1, 2]}}
    """
    game = get_arena_game(game_id)
    if not game:
        abort(404, description="Game not found")

    current_player = game.state.current_player()
    if not isinstance(current_player, WebLLMPlayer):
        abort(400, description="Current player is not an LLM")

    if not request.is_json:
        abort(400, description="Request must be JSON")

    # Get valid actions
    valid_actions = [
        {"action_type": a.action_type.name, "value": a.value}
        for a in game.playable_actions
    ]

    # Parse action from response or direct action
    if "response" in request.json:
        # Parse LLM response text
        try:
            parsed = action_parser.parse(request.json["response"], valid_actions)
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e),
                "valid_actions": valid_actions,
            }), 400
    elif "action" in request.json:
        parsed = request.json["action"]
    else:
        abort(400, description="Must provide 'response' or 'action'")

    # Find matching game action
    action_type = parsed.get("action_type")
    value = parsed.get("value")

    matching_action = None
    for action in game.playable_actions:
        if action.action_type.name == action_type:
            # Check value match
            if value is None and action.value is None:
                matching_action = action
                break
            elif value == action.value:
                matching_action = action
                break
            elif isinstance(value, list) and isinstance(action.value, (list, tuple)):
                if list(value) == list(action.value) or tuple(sorted(value)) == tuple(sorted(action.value)):
                    matching_action = action
                    break

    if not matching_action:
        return jsonify({
            "success": False,
            "error": f"Action not valid: {action_type} with value {value}",
            "valid_actions": valid_actions,
        }), 400

    # Record and execute action
    record_action(game, matching_action, current_player.color.name)
    game.execute(matching_action)
    save_arena_game(game)

    # Auto-roll if now needed (e.g., turn ended, next player needs to roll)
    roll_info = auto_roll_if_needed(game)
    if roll_info:
        save_arena_game(game)

    # Return updated state
    return jsonify({
        "success": True,
        "action_executed": {
            "action_type": matching_action.action_type.name,
            "value": matching_action.value,
        },
        "dice_rolled": roll_info,
        "game_state": get_arena_game_state(game_id).json,
    })


@bp.route("/games/<game_id>/bot-action", methods=["POST"])
def execute_bot_action(game_id: str):
    """Execute next action for a bot player (Random/Catanatron)."""
    game = get_arena_game(game_id)
    if not game:
        abort(404, description="Game not found")

    current_player = game.state.current_player()
    if isinstance(current_player, WebLLMPlayer):
        abort(400, description="Current player is LLM - use /llm-action endpoint")

    if game.winning_color():
        return jsonify({"message": "Game already finished"})

    # Execute bot action
    game.play_tick()
    save_arena_game(game)

    return jsonify({
        "success": True,
        "game_id": game_id,
    })


@bp.route("/games/<game_id>/auto-play", methods=["POST"])
def auto_play_until_llm(game_id: str):
    """Auto-play bot turns until it's an LLM's turn or game ends."""
    game = get_arena_game(game_id)
    if not game:
        abort(404, description="Game not found")

    actions_taken = 0
    max_actions = 100  # Safety limit

    while actions_taken < max_actions:
        if game.winning_color():
            break

        current_player = game.state.current_player()

        # Auto-roll if needed
        roll_info = auto_roll_if_needed(game)
        if roll_info:
            actions_taken += 1
            continue

        # Check if it's an LLM's turn (after rolling)
        if isinstance(current_player, WebLLMPlayer):
            break

        # Execute bot action
        playable_actions = game.playable_actions
        if playable_actions:
            action = current_player.decide(game, playable_actions)
            record_action(game, action, current_player.color.name)
            game.execute(action)
            actions_taken += 1

    save_arena_game(game)

    # Return current state
    current_player = game.state.current_player()
    response = {
        "actions_taken": actions_taken,
        "is_finished": game.winning_color() is not None,
        "current_player": current_player.color.name,
    }

    if isinstance(current_player, WebLLMPlayer):
        # Auto-roll if needed for LLM
        roll_info = auto_roll_if_needed(game)
        if roll_info:
            save_arena_game(game)
            response["dice_rolled"] = roll_info

        response["llm_prompt"] = generate_llm_prompt(game, current_player)
        response["waiting_for_llm"] = True

    return jsonify(response)


@bp.route("/games", methods=["GET"])
def list_arena_games():
    """List all active arena games."""
    games = []
    for game_id, game in ARENA_GAMES.items():
        games.append({
            "game_id": game_id,
            "players": [
                {
                    "color": p.color.name,
                    "type": type(p).__name__,
                    "name": getattr(p, "name", None),
                }
                for p in game.state.players
            ],
            "is_finished": game.winning_color() is not None,
            "winner": game.winning_color().name if game.winning_color() else None,
        })
    return jsonify({"games": games})
