"""
Arena Server - FastAPI backend for running LLM games.

Provides:
- REST endpoints to start games
- WebSocket for real-time game updates
- Game state management
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Game imports
from arena.config import create_config
from arena.registry import get_game_class
from games.codenames.game import CodenamesGame
from games.codenames.state_adapter import CodenamesStateAdapter
from arena.llm.llm_agent import LLMAgent
from core.agent import RandomAgent

# Catan imports
from catanatron.game import Game as CatanGame
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.json import GameEncoder
from games.catan.state_adapter import CatanStateAdapter

app = FastAPI(title="Game Arena API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active games
active_games: Dict[str, Dict[str, Any]] = {}

# WebSocket connections per game
game_connections: Dict[str, List[WebSocket]] = {}


# ============================================
# Request/Response Models
# ============================================

class PlayerConfig(BaseModel):
    id: Optional[str] = None
    team: Optional[str] = None
    role: Optional[str] = None
    type: str = "llm"  # llm, random, manual
    model: Optional[str] = "gpt-4o"
    color: Optional[str] = None


class GameConfig(BaseModel):
    word_list: str = "standard"
    max_turns: int = 50
    map_type: str = "BASE"
    vps_to_win: int = 10


class StartGameRequest(BaseModel):
    players: List[PlayerConfig]
    config: Optional[GameConfig] = None


class StartGameResponse(BaseModel):
    game_id: str
    game_type: str
    status: str
    message: str


# ============================================
# REST Endpoints
# ============================================

@app.get("/")
async def root():
    return {"message": "Game Arena API", "version": "1.0.0"}


@app.get("/api/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/arena/codenames/start", response_model=StartGameResponse)
async def start_codenames_game(request: StartGameRequest):
    """Start a new Codenames game with LLM players."""
    game_id = str(uuid.uuid4())[:8]

    # Parse player configs
    red_spymaster = None
    red_operatives = []
    blue_spymaster = None
    blue_operatives = []
    agents = {}

    for i, p in enumerate(request.players):
        player_id = p.id or f"player_{i}"

        # Assign team/role
        if p.team == "red":
            if p.role == "spymaster":
                red_spymaster = player_id
            else:
                red_operatives.append(player_id)
        elif p.team == "blue":
            if p.role == "spymaster":
                blue_spymaster = player_id
            else:
                blue_operatives.append(player_id)

        # Create agent
        if p.type == "llm":
            agents[player_id] = LLMAgent(
                agent_id=player_id,
                model=p.model or "gpt-4o",
            )
        else:
            agents[player_id] = RandomAgent(agent_id=player_id)

    # Create game
    game = CodenamesGame(
        red_spymaster=red_spymaster or "red_spymaster",
        red_operatives=red_operatives or ["red_operative"],
        blue_spymaster=blue_spymaster or "blue_spymaster",
        blue_operatives=blue_operatives or ["blue_operative"],
    )

    # Store game state
    active_games[game_id] = {
        "game": game,
        "agents": agents,
        "state_adapter": CodenamesStateAdapter(),
        "players": [p.model_dump() for p in request.players],
        "config": request.config.model_dump() if request.config else {},
        "created_at": datetime.utcnow().isoformat(),
        "status": "created",
        "turn_history": [],
    }

    game_connections[game_id] = []

    return StartGameResponse(
        game_id=game_id,
        game_type="codenames",
        status="created",
        message=f"Game {game_id} created. Connect to WebSocket to start.",
    )


@app.post("/api/arena/catan/start", response_model=StartGameResponse)
async def start_catan_game(request: StartGameRequest):
    """Start a new Catan game with LLM players."""
    game_id = str(uuid.uuid4())[:8]

    # Create Catan players
    players = []
    agents = {}
    color_list = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]

    for i, p in enumerate(request.players[:4]):  # Max 4 players for Catan
        color = color_list[i]
        player_id = p.id or f"player_{color.value}"

        # Create catanatron player (all LLM players are treated as "non-bot" for the game engine)
        if p.type == "llm":
            catan_player = ValueFunctionPlayer(color, is_bot=False)
            agents[player_id] = LLMAgent(
                agent_id=player_id,
                model=p.model or "gpt-4o",
            )
        else:
            catan_player = RandomPlayer(color)
            agents[player_id] = RandomAgent(agent_id=player_id)

        players.append(catan_player)

    # Create Catan game
    catan_game = CatanGame(players=players)

    # Store game state
    active_games[game_id] = {
        "game": catan_game,
        "game_type": "catan",
        "agents": agents,
        "state_adapter": CatanStateAdapter(),
        "players": [p.model_dump() for p in request.players[:4]],
        "config": request.config.model_dump() if request.config else {},
        "created_at": datetime.utcnow().isoformat(),
        "status": "created",
        "turn_history": [],
    }

    game_connections[game_id] = []

    return StartGameResponse(
        game_id=game_id,
        game_type="catan",
        status="created",
        message=f"Catan game {game_id} created. Connect to WebSocket to start.",
    )


@app.get("/api/arena/games/{game_id}")
async def get_game_state(game_id: str):
    """Get current game state."""
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")

    game_data = active_games[game_id]
    game = game_data["game"]

    return {
        "game_id": game_id,
        "game_type": "codenames",
        "status": game_data["status"],
        "public_state": game.get_public_state(),
        "is_game_over": game.is_over(),
        "winner": game.get_winner(),
    }


@app.get("/api/arena/games")
async def list_games():
    """List all active games."""
    return {
        "games": [
            {
                "game_id": gid,
                "game_type": "codenames",
                "status": gdata["status"],
                "created_at": gdata["created_at"],
            }
            for gid, gdata in active_games.items()
        ]
    }


# ============================================
# WebSocket for Real-time Updates
# ============================================

@app.websocket("/ws/arena/{game_id}")
async def websocket_game(websocket: WebSocket, game_id: str):
    """
    WebSocket endpoint for real-time game updates.

    Connect to this after starting a game to receive turn updates.
    Send {"action": "start"} to begin the game loop.
    """
    await websocket.accept()

    if game_id not in active_games:
        await websocket.send_json({"error": "Game not found"})
        await websocket.close()
        return

    game_connections[game_id].append(websocket)

    try:
        # Send initial state
        game_data = active_games[game_id]
        game = game_data["game"]
        game_type = game_data.get("game_type", "codenames")

        # Get initial state based on game type
        if game_type == "catan":
            initial_state = json.loads(json.dumps(game, cls=GameEncoder))
        else:
            initial_state = game.get_public_state()

        await websocket.send_json({
            "type": "connected",
            "game_id": game_id,
            "game_type": game_type,
            "initial_state": initial_state,
            "players": game_data["players"],
        })

        # Wait for commands
        while True:
            data = await websocket.receive_json()

            if data.get("action") == "start":
                # Run the game loop based on game type
                if game_type == "catan":
                    await run_catan_game_loop(game_id, websocket)
                else:
                    await run_game_loop(game_id, websocket)
            elif data.get("action") == "submit_action":
                # Handle manual action submission
                if game_type == "catan":
                    await handle_manual_action(game_id, data, websocket)
                else:
                    # TODO: Handle codenames manual actions
                    pass
            elif data.get("action") == "state":
                # Return current state
                if game_type == "catan":
                    current_state = json.loads(json.dumps(game, cls=GameEncoder))
                else:
                    current_state = game.get_public_state()
                await websocket.send_json({
                    "type": "state",
                    "public_state": current_state,
                })

    except WebSocketDisconnect:
        game_connections[game_id].remove(websocket)
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
        game_connections[game_id].remove(websocket)


async def run_game_loop(game_id: str, websocket: WebSocket):
    """Run the game loop, sending updates via WebSocket."""
    game_data = active_games[game_id]
    game = game_data["game"]
    agents = game_data["agents"]
    state_adapter = game_data["state_adapter"]

    game_data["status"] = "running"

    await broadcast_to_game(game_id, {
        "type": "game_started",
        "game_id": game_id,
    })

    turn_number = 0
    max_turns = game_data["config"].get("max_turns", 50)

    while not game.is_over() and turn_number < max_turns:
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

        # Notify: thinking
        await broadcast_to_game(game_id, {
            "type": "turn_start",
            "turn_number": turn_number,
            "current_player": current_player_id,
            "current_role": current_role,
            "current_team": public_state.get("current_team"),
            "public_state": public_state,
        })

        # Get agent
        agent = agents.get(current_player_id)
        if agent is None:
            agent = RandomAgent(agent_id=current_player_id)

        # Generate prompt
        prompt = state_adapter.state_to_prompt(
            public_state=public_state,
            private_state=private_state,
            valid_actions=valid_actions,
        )

        system_prompt = state_adapter.format_system_prompt(
            role=current_role,
            team=private_state.get("team"),
            clue=public_state.get("current_clue", {}).get("word") if public_state.get("current_clue") else None,
            number=public_state.get("current_clue", {}).get("number") if public_state.get("current_clue") else None,
        )

        # Notify: LLM thinking
        await broadcast_to_game(game_id, {
            "type": "llm_thinking",
            "player_id": current_player_id,
            "model": agent.get_metadata().get("model", "unknown"),
        })

        # Small delay for UI
        await asyncio.sleep(0.5)

        # Get decision
        try:
            decision = agent.decide(
                prompt=prompt,
                valid_actions=valid_actions,
                system_prompt=system_prompt,
                game_type="codenames",
                role=current_role,
                turn_number=turn_number,
            )
        except Exception as e:
            decision = {
                "action": valid_actions[0] if valid_actions else {},
                "reasoning": f"Error: {e}",
                "raw_output": "",
                "metadata": {"error": str(e)},
            }

        action = decision.get("action", {})
        reasoning = decision.get("reasoning", "")

        # Notify: decision made
        await broadcast_to_game(game_id, {
            "type": "llm_decision",
            "player_id": current_player_id,
            "action": action,
            "reasoning": reasoning,
            "metadata": decision.get("metadata", {}),
        })

        await asyncio.sleep(0.5)

        # Execute action
        action_result, game_over = game.step(action)

        # Notify: action executed
        await broadcast_to_game(game_id, {
            "type": "action_executed",
            "turn_number": turn_number,
            "player_id": current_player_id,
            "action": action,
            "result": action_result,
            "public_state": game.get_public_state(),
            "game_over": game_over,
        })

        # Record turn
        game_data["turn_history"].append({
            "turn_number": turn_number,
            "player_id": current_player_id,
            "role": current_role,
            "action": action,
            "result": action_result,
            "reasoning": reasoning,
        })

        # Delay between turns
        await asyncio.sleep(1)

    # Game finished
    game_data["status"] = "finished"
    winner = game.get_winner()
    scores = game.get_scores()

    await broadcast_to_game(game_id, {
        "type": "game_finished",
        "game_id": game_id,
        "winner": winner,
        "scores": scores,
        "total_turns": turn_number,
        "termination_reason": "victory" if winner else "turn_limit",
    })

    # Save game log
    save_game_log(game_id, game_data)


async def run_catan_game_loop(game_id: str, websocket: WebSocket):
    """Run the Catan game loop, sending updates via WebSocket.

    Supports manual mode where users copy prompts and paste LLM responses.
    """
    game_data = active_games[game_id]
    game = game_data["game"]
    agents = game_data["agents"]
    state_adapter = game_data["state_adapter"]

    game_data["status"] = "running"

    await broadcast_to_game(game_id, {
        "type": "game_started",
        "game_id": game_id,
        "game_type": "catan",
    })

    turn_number = 0
    max_turns = game_data["config"].get("max_turns", 200)

    while game.winning_color() is None and turn_number < max_turns:
        turn_number += 1

        # Get current player
        current_player = game.state.current_player()
        current_color = current_player.color.value
        player_id = f"player_{current_color}"

        # Get game state in UI format
        game_state = json.loads(json.dumps(game, cls=GameEncoder))

        # Build public state for prompt generation
        public_state = {
            "board": {
                "tiles": game_state.get("tiles", []),
                "buildings": [],
                "roads": [],
            },
            "phase": "INITIAL_BUILD" if game.state.is_initial_build_phase else "MAIN",
            "turn_number": turn_number,
            "current_player": player_id,
            "robber_coordinate": game_state.get("robber_coordinate"),
            "player_summaries": {},
        }

        # Get private state (player's hand)
        player_state = game.state.player_state
        private_state = {
            "hand_resources": player_state,
            "hand_dev_cards": {},
            "actual_victory_points": 0,
        }

        # Get valid actions
        valid_actions = state_adapter.format_valid_actions_for_prompt(game.playable_actions)

        # Generate prompt
        prompt = state_adapter.state_to_prompt(
            public_state=public_state,
            private_state=private_state,
            valid_actions=valid_actions,
        )

        system_prompt = state_adapter.format_system_prompt()

        # Notify: turn start with prompt for manual play
        await broadcast_to_game(game_id, {
            "type": "turn_start",
            "turn_number": turn_number,
            "current_player": player_id,
            "current_color": current_color,
            "game_state": game_state,
            "is_initial_build_phase": game.state.is_initial_build_phase,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "valid_actions": valid_actions,
        })

        # Check if this is a manual player - wait for user input
        agent = agents.get(player_id)
        is_manual = agent is None or (hasattr(agent, 'agent_type') and agent.agent_type == 'manual')

        # For manual mode, we need all players to be manual
        # Check the player config
        player_configs = game_data.get("players", [])
        player_index = ["RED", "BLUE", "ORANGE", "WHITE"].index(current_color) if current_color in ["RED", "BLUE", "ORANGE", "WHITE"] else 0
        if player_index < len(player_configs):
            player_config = player_configs[player_index]
            is_manual = player_config.get("type") == "manual"

        if is_manual:
            # Wait for user to submit action via WebSocket
            await broadcast_to_game(game_id, {
                "type": "waiting_for_input",
                "player_id": player_id,
                "current_color": current_color,
                "message": "Copy the prompt above, send to your LLM, then paste the JSON response below.",
            })

            # Store that we're waiting for this player
            game_data["waiting_for"] = player_id
            game_data["current_turn"] = turn_number
            return  # Exit loop - will continue when user submits action

        elif agent is None or isinstance(agent, RandomAgent):
            # Use the game's built-in play_tick for random/bot players
            await asyncio.sleep(0.3)
            game.play_tick()
        else:
            # LLM agent - call API
            await broadcast_to_game(game_id, {
                "type": "llm_thinking",
                "player_id": player_id,
                "model": agent.get_metadata().get("model", "unknown"),
            })

            await asyncio.sleep(0.5)

            try:
                decision = agent.decide(
                    prompt=prompt,
                    valid_actions=valid_actions,
                    system_prompt=system_prompt,
                    game_type="catan",
                    role="player",
                    turn_number=turn_number,
                )
            except Exception as e:
                decision = {"action": None, "reasoning": f"Error: {e}"}
                game.play_tick()

            if decision.get("action"):
                action = decision.get("action", {})
                reasoning = decision.get("reasoning", "")

                await broadcast_to_game(game_id, {
                    "type": "llm_decision",
                    "player_id": player_id,
                    "action": action,
                    "reasoning": reasoning,
                })

                await asyncio.sleep(0.5)

                try:
                    from catanatron.json import action_from_json
                    action_type = action.get("action_type")
                    value = action.get("value")
                    catan_action = action_from_json([current_color, action_type, value])
                    game.execute(catan_action)
                except Exception as e:
                    game.play_tick()
            else:
                game.play_tick()

        # Get updated game state
        game_state = json.loads(json.dumps(game, cls=GameEncoder))

        # Notify: action executed
        await broadcast_to_game(game_id, {
            "type": "action_executed",
            "turn_number": turn_number,
            "player_id": player_id,
            "game_state": game_state,
            "game_over": game.winning_color() is not None,
        })

        # Record turn
        game_data["turn_history"].append({
            "turn_number": turn_number,
            "player_id": player_id,
            "color": current_color,
        })

        await asyncio.sleep(0.5)

    # Game finished
    game_data["status"] = "finished"
    winner = game.winning_color()

    await broadcast_to_game(game_id, {
        "type": "game_finished",
        "game_id": game_id,
        "game_type": "catan",
        "winner": winner.value if winner else None,
        "total_turns": turn_number,
        "termination_reason": "victory" if winner else "turn_limit",
    })


async def handle_manual_action(game_id: str, action_data: Dict[str, Any], websocket: WebSocket):
    """Handle a manually submitted action from the user."""
    if game_id not in active_games:
        await websocket.send_json({"type": "error", "message": "Game not found"})
        return

    game_data = active_games[game_id]
    game = game_data["game"]

    if game_data.get("waiting_for") is None:
        await websocket.send_json({"type": "error", "message": "Not waiting for input"})
        return

    player_id = game_data["waiting_for"]
    current_color = player_id.replace("player_", "")

    try:
        # Parse the action from user input
        # The LLM response has action_type and value at the top level
        # action_data contains: {"action": "submit_action", "action_type": "...", "value": ..., "rationale": "..."}
        action = {
            "action_type": action_data.get("action_type"),
            "value": action_data.get("value"),
        }
        reasoning = action_data.get("reasoning", action_data.get("rationale", ""))

        # Broadcast the decision
        await broadcast_to_game(game_id, {
            "type": "llm_decision",
            "player_id": player_id,
            "action": action,
            "reasoning": reasoning,
        })

        await asyncio.sleep(0.3)

        # Execute the action
        from catanatron.json import action_from_json
        action_type = action.get("action_type")
        value = action.get("value")

        catan_action = action_from_json([current_color, action_type, value])
        game.execute(catan_action)

        # Clear waiting state
        game_data["waiting_for"] = None

        # Get updated game state
        game_state = json.loads(json.dumps(game, cls=GameEncoder))

        # Notify: action executed
        await broadcast_to_game(game_id, {
            "type": "action_executed",
            "turn_number": game_data.get("current_turn", 0),
            "player_id": player_id,
            "game_state": game_state,
            "game_over": game.winning_color() is not None,
        })

        # Continue the game loop
        if game.winning_color() is None:
            await asyncio.sleep(0.5)
            await run_catan_game_loop(game_id, websocket)

    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": f"Invalid action: {str(e)}",
        })


async def broadcast_to_game(game_id: str, message: Dict[str, Any]):
    """Broadcast message to all WebSocket connections for a game."""
    if game_id not in game_connections:
        return

    disconnected = []
    for ws in game_connections[game_id]:
        try:
            await ws.send_json(message)
        except:
            disconnected.append(ws)

    # Clean up disconnected
    for ws in disconnected:
        game_connections[game_id].remove(ws)


def save_game_log(game_id: str, game_data: Dict[str, Any]):
    """Save game log to file."""
    log_dir = Path("./game_logs")
    log_dir.mkdir(exist_ok=True)

    game = game_data["game"]

    log = {
        "game_id": game_id,
        "game_type": "codenames",
        "created_at": game_data["created_at"],
        "finished_at": datetime.utcnow().isoformat(),
        "players": game_data["players"],
        "config": game_data["config"],
        "initial_state": {
            "grid": game.grid,
            "card_types": {w: ct.value for w, ct in game.card_types.items()},
        },
        "result": {
            "winner": game.get_winner(),
            "scores": game.get_scores(),
            "total_turns": len(game_data["turn_history"]),
        },
        "turns": game_data["turn_history"],
    }

    log_path = log_dir / f"codenames_{game_id}.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, default=str)

    print(f"Game log saved: {log_path}")


# ============================================
# Run Server
# ============================================

if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="Game Arena Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    print(f"""
╔═══════════════════════════════════════════════════════════╗
║              GAME ARENA SERVER                            ║
╠═══════════════════════════════════════════════════════════╣
║  Starting server at http://{args.host}:{args.port}                   ║
║                                                           ║
║  Endpoints:                                               ║
║    POST /api/arena/codenames/start - Start Codenames game ║
║    GET  /api/arena/games           - List active games    ║
║    WS   /ws/arena/{{game_id}}        - Game WebSocket      ║
║                                                           ║
║  Set API keys:                                            ║
║    export OPENAI_API_KEY=sk-...                          ║
║    export ANTHROPIC_API_KEY=sk-ant-...                   ║
╚═══════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "arena.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
