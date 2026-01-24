# Catan-Arena: Technical Design Document

**Version:** 1.0.0-MVP
**Author:** Principal Software Architect
**Date:** January 2026
**Status:** Draft for Review

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [The Data Contract (JSON Schema)](#3-the-data-contract-json-schema)
4. [Agent-Environment Interface](#4-agent-environment-interface)
5. [The React Replay Viewer](#5-the-react-replay-viewer)
6. [Extensibility Plan](#6-extensibility-plan)
7. [Implementation Roadmap](#7-implementation-roadmap)

---

## 1. Executive Summary

**Catan-Arena** is a scalable benchmarking platform where Large Language Models (LLMs) compete in the board game Settlers of Catan. The system leverages the existing **Catanatron** Python library as the game engine, wrapped in a **PettingZoo AEC** (Actor-Environment-Cycle) interface for multi-agent coordination, orchestrated via **LangGraph** for stateful agent management, with a **React** frontend for game replay visualization.

### Core Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Separation of Concerns** | Game logic (Catanatron) is isolated from agent logic (LangGraph) and UI (React) |
| **State Immutability** | All game states are snapshots; mutations create new states |
| **Game-Agnostic Core** | LangGraph orchestration and React viewer work with any game via adapters |
| **Observable Decisions** | Every LLM decision includes Chain-of-Thought reasoning for analysis |

### Technology Stack

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CATAN-ARENA MVP                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  Frontend        │  React 19 + TypeScript + MUI                             │
│  Orchestration   │  LangGraph (Python)                                      │
│  Multi-Agent     │  PettingZoo AEC API                                      │
│  Game Engine     │  Catanatron (Python)                                     │
│  Storage         │  File-based JSON logs (no SQL for MVP)                   │
│  LLM Integration │  LangChain (Claude, GPT-4, Gemini, etc.)                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Architecture Overview

### 2.1 Why LangGraph?

LangGraph is the orchestration backbone of Catan-Arena for three critical reasons:

#### 2.1.1 State Snapshot Persistence

LangGraph's **checkpointer** feature allows us to serialize the entire orchestration state at any point:

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

# Create checkpointer for state persistence
checkpointer = MemorySaver()

# Graph with checkpointing enabled
graph = workflow.compile(checkpointer=checkpointer)

# Execute with thread_id for state tracking
config = {"configurable": {"thread_id": "game_abc123"}}
result = graph.invoke(initial_state, config)

# Later: Resume from any checkpoint
state_at_turn_15 = graph.get_state(config, checkpoint_id="turn_15")
```

**Use Cases Enabled:**
- **Pause/Resume Games**: Save a 4-player LLM match and resume hours later
- **Turn-by-Turn Debugging**: Inspect exactly what any LLM "saw" at turn N
- **A/B Testing**: Fork from turn 15 and replay with different LLM configurations
- **Human-in-the-Loop**: Pause game, let human override one LLM's decision, continue

#### 2.1.2 Cyclic Graph for Turn-Based Games

Unlike DAG-based pipelines, LangGraph supports **cycles**, which naturally model turn-based games:

```
                    ┌─────────────────────────────────────┐
                    │                                     │
                    ▼                                     │
┌──────────────────────────────────────────────────────────────────────────┐
│                         LANGGRAPH GAME LOOP                              │
│                                                                          │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐               │
│   │ Get_Board   │────▶│ LLM_Decide  │────▶│ Parse_Action│               │
│   │   _State    │     │             │     │             │               │
│   └─────────────┘     └─────────────┘     └─────────────┘               │
│          ▲                                       │                       │
│          │                                       ▼                       │
│          │            ┌─────────────┐     ┌─────────────┐               │
│          │            │ Check_Game  │◀────│ Update_Env  │               │
│          │            │   _Over     │     │             │               │
│          │            └─────────────┘     └─────────────┘               │
│          │                   │                                          │
│          │                   │ game_active=True                         │
│          └───────────────────┘                                          │
│                              │ game_over=True                           │
│                              ▼                                          │
│                       ┌─────────────┐                                   │
│                       │ Finalize    │                                   │
│                       │   _Game     │                                   │
│                       └─────────────┘                                   │
└──────────────────────────────────────────────────────────────────────────┘
```

#### 2.1.3 Built-in Retry and Error Handling

LangGraph provides structured error handling for LLM failures:

```python
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableConfig

# Automatic retry with backoff for transient LLM API failures
@retry(max_attempts=3, backoff_factor=2.0)
async def llm_decide_node(state: GameState, config: RunnableConfig):
    # If LLM returns invalid action, we can route to retry logic
    ...
```

### 2.2 LangGraph Node Definitions

```python
from typing import TypedDict, Literal, List, Optional, Annotated
from langgraph.graph import StateGraph, END
from operator import add

# ═══════════════════════════════════════════════════════════════════════════
# SHARED STATE SCHEMA
# ═══════════════════════════════════════════════════════════════════════════

class AgentDecision(TypedDict):
    """Single agent's decision with reasoning"""
    agent_id: str
    model_name: str
    reasoning: str  # Chain-of-thought
    raw_output: str
    parsed_action: dict
    latency_ms: int
    token_count: int

class TurnRecord(TypedDict):
    """Complete record of one game turn"""
    turn_number: int
    current_player: str
    public_state: dict
    private_states: dict[str, dict]  # Keyed by player_id
    valid_actions: List[dict]
    decision: AgentDecision
    action_result: dict
    timestamp_utc: str

class ArenaGameState(TypedDict):
    """LangGraph shared state for the entire game"""
    # Game identification
    game_id: str
    game_type: Literal["catan", "chess", "diplomacy"]

    # Player configuration
    players: List[dict]  # [{id, model, color, is_human}, ...]
    current_player_index: int

    # Game state (from PettingZoo env)
    env_state: dict  # Serialized PettingZoo observation
    turn_number: int
    phase: Literal["setup", "main", "finished"]

    # Accumulating history (Annotated for append-only updates)
    turn_history: Annotated[List[TurnRecord], add]

    # Terminal state
    winner: Optional[str]
    final_scores: Optional[dict[str, int]]
    termination_reason: Optional[str]

# ═══════════════════════════════════════════════════════════════════════════
# NODE IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_board_state(state: ArenaGameState) -> dict:
    """
    Node: Get_Board_State

    Extracts the current game state from PettingZoo environment and
    prepares both public and private views for the current player.
    """
    env = state["_env"]  # PettingZoo env stored in state
    current_agent = env.agent_selection

    # Get observation for current agent (includes action mask)
    observation, reward, termination, truncation, info = env.last()

    # Separate public vs private state
    public_state = extract_public_state(env, observation)
    private_state = extract_private_state(env, observation, current_agent)

    return {
        "env_state": {
            "public": public_state,
            "private": {current_agent: private_state},
            "valid_actions": info.get("valid_actions", []),
            "action_mask": info.get("action_mask", None),
        },
        "current_player_index": state["players"].index(
            next(p for p in state["players"] if p["id"] == current_agent)
        ),
    }


def llm_decide(state: ArenaGameState) -> dict:
    """
    Node: LLM_Decide

    Invokes the appropriate LLM for the current player, passing the
    game state through the StateAdapter to generate a text prompt.
    """
    import time
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI

    current_player = state["players"][state["current_player_index"]]
    model_name = current_player["model"]

    # Select LLM based on player configuration
    llm = get_llm_for_model(model_name)

    # Generate prompt via StateAdapter (game-specific)
    adapter = get_state_adapter(state["game_type"])
    prompt = adapter.state_to_prompt(
        public_state=state["env_state"]["public"],
        private_state=state["env_state"]["private"][current_player["id"]],
        valid_actions=state["env_state"]["valid_actions"],
        turn_history=state["turn_history"][-10:],  # Last 10 turns for context
    )

    # Invoke LLM with structured output
    start_time = time.time()
    response = llm.invoke(prompt)
    latency_ms = int((time.time() - start_time) * 1000)

    return {
        "_pending_decision": {
            "agent_id": current_player["id"],
            "model_name": model_name,
            "reasoning": extract_reasoning(response),
            "raw_output": response.content,
            "latency_ms": latency_ms,
            "token_count": response.usage_metadata.get("total_tokens", 0),
        }
    }


def parse_action(state: ArenaGameState) -> dict:
    """
    Node: Parse_Action

    Parses the LLM's raw output into a valid game action.
    Uses the ActionParser which is game-specific.
    """
    decision = state["_pending_decision"]
    parser = get_action_parser(state["game_type"])

    try:
        parsed_action = parser.parse(
            raw_output=decision["raw_output"],
            valid_actions=state["env_state"]["valid_actions"],
        )
        decision["parsed_action"] = parsed_action
        decision["parse_success"] = True
    except ActionParseError as e:
        # Fallback: select random valid action
        decision["parsed_action"] = random.choice(state["env_state"]["valid_actions"])
        decision["parse_success"] = False
        decision["parse_error"] = str(e)

    return {"_pending_decision": decision}


def update_env(state: ArenaGameState) -> dict:
    """
    Node: Update_Env

    Applies the parsed action to the PettingZoo environment
    and advances to the next player.
    """
    env = state["_env"]
    action = state["_pending_decision"]["parsed_action"]

    # Convert parsed action to PettingZoo action space
    action_int = action_to_env_space(action, state["game_type"])

    # Execute action in environment
    env.step(action_int)

    # Get result of action
    observation, reward, termination, truncation, info = env.last()

    # Create turn record
    turn_record: TurnRecord = {
        "turn_number": state["turn_number"],
        "current_player": state["players"][state["current_player_index"]]["id"],
        "public_state": state["env_state"]["public"],
        "private_states": state["env_state"]["private"],
        "valid_actions": state["env_state"]["valid_actions"],
        "decision": state["_pending_decision"],
        "action_result": {
            "reward": reward,
            "info": info,
        },
        "timestamp_utc": datetime.utcnow().isoformat(),
    }

    return {
        "turn_number": state["turn_number"] + 1,
        "turn_history": [turn_record],  # Appended via Annotated[..., add]
    }


def check_game_over(state: ArenaGameState) -> Literal["continue", "finished"]:
    """
    Conditional Edge: Determines if game should continue or end.
    """
    env = state["_env"]

    # Check for winner
    if hasattr(env, 'unwrapped'):
        game = env.unwrapped.game
        if game.winning_color() is not None:
            return "finished"

    # Check for truncation (turn limit)
    if state["turn_number"] >= 1000:
        return "finished"

    return "continue"


def finalize_game(state: ArenaGameState) -> dict:
    """
    Node: Finalize_Game

    Records final scores, determines winner, and saves game log.
    """
    env = state["_env"]
    game = env.unwrapped.game

    # Extract final scores
    final_scores = {}
    for player in state["players"]:
        color = player_id_to_color(player["id"])
        vps = get_actual_victory_points(game.state, color)
        final_scores[player["id"]] = vps

    winner_color = game.winning_color()
    winner_id = color_to_player_id(winner_color, state["players"]) if winner_color else None

    return {
        "phase": "finished",
        "winner": winner_id,
        "final_scores": final_scores,
        "termination_reason": "victory" if winner_id else "turn_limit",
    }


# ═══════════════════════════════════════════════════════════════════════════
# GRAPH ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════

def build_arena_graph() -> StateGraph:
    """Constructs the LangGraph workflow for game execution."""

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

    # Conditional edge: continue or end
    workflow.add_conditional_edges(
        "update_env",
        check_game_over,
        {
            "continue": "get_board_state",  # Loop back
            "finished": "finalize_game",    # Exit loop
        }
    )

    workflow.add_edge("finalize_game", END)

    return workflow


# ═══════════════════════════════════════════════════════════════════════════
# EXECUTION ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════

async def run_arena_game(
    game_type: str,
    player_configs: List[dict],
    checkpoint_dir: str = "./checkpoints"
) -> ArenaGameState:
    """
    Main entrypoint to run a Catan-Arena game.

    Args:
        game_type: "catan", "chess", etc.
        player_configs: List of {model: "claude-3-opus", color: "RED", ...}
        checkpoint_dir: Directory for state snapshots

    Returns:
        Final game state with complete history
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    # Initialize PettingZoo environment
    env = create_pettingzoo_env(game_type, player_configs)
    env.reset()

    # Build graph with checkpointing
    workflow = build_arena_graph()
    checkpointer = SqliteSaver.from_conn_string(f"{checkpoint_dir}/games.db")
    graph = workflow.compile(checkpointer=checkpointer)

    # Initial state
    initial_state: ArenaGameState = {
        "game_id": str(uuid.uuid4()),
        "game_type": game_type,
        "players": player_configs,
        "current_player_index": 0,
        "env_state": {},
        "turn_number": 0,
        "phase": "setup",
        "turn_history": [],
        "winner": None,
        "final_scores": None,
        "termination_reason": None,
        "_env": env,  # PettingZoo env (not serialized)
    }

    # Execute game
    config = {"configurable": {"thread_id": initial_state["game_id"]}}
    final_state = await graph.ainvoke(initial_state, config)

    # Save final game log
    save_game_log(final_state, f"./logs/{final_state['game_id']}.json")

    return final_state
```

### 2.3 The Catanatron-PettingZoo Bridge

The existing Catanatron library provides a Gymnasium environment (`CatanatronEnv`), but for multi-agent LLM competition, we need **PettingZoo's AEC (Actor-Environment-Cycle)** API which properly handles:

1. **Turn-based multi-agent**: Each agent acts in sequence, not simultaneously
2. **Per-agent observations**: Each agent sees only their allowed information
3. **Action masking**: Invalid actions are masked per-agent

```python
# catan_arena/envs/catan_pettingzoo.py

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from gymnasium import spaces
import numpy as np

from catanatron.game import Game
from catanatron.models.player import Player, Color
from catanatron.models.enums import ActionType, Action
from catanatron.state_functions import (
    get_player_freqdeck,
    get_dev_cards_in_hand,
    get_actual_victory_points,
    player_num_resource_cards,
)


class CatanAECEnv(AECEnv):
    """
    PettingZoo AEC wrapper for Catanatron.

    Converts the single-threaded Catanatron game engine into a multi-agent
    environment where each agent (LLM) takes turns making decisions.

    Key Features:
    - 2-4 player support
    - Per-agent observation spaces (public + private views)
    - Action masking for valid moves only
    - Full game state serialization for replay
    """

    metadata = {
        "render_modes": ["human", "json"],
        "name": "catan_arena_v0",
        "is_parallelizable": False,
    }

    def __init__(
        self,
        num_players: int = 4,
        map_type: str = "BASE",
        vps_to_win: int = 10,
        max_turns: int = 1000,
        render_mode: str = None,
    ):
        super().__init__()

        self.num_players = num_players
        self.map_type = map_type
        self.vps_to_win = vps_to_win
        self.max_turns = max_turns
        self.render_mode = render_mode

        # Define agents (colors)
        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self.agent_colors = {
            "player_0": Color.RED,
            "player_1": Color.BLUE,
            "player_2": Color.ORANGE,
            "player_3": Color.WHITE,
        }

        # Action space: matches existing CatanatronEnv
        # 368 discrete actions covering all possible moves
        self._action_space = spaces.Discrete(368)

        # Observation space: dict with public and private components
        self._observation_space = spaces.Dict({
            # Public state (visible to all)
            "public": spaces.Dict({
                "board_tensor": spaces.Box(0, 1, (21, 11, 32), dtype=np.float32),
                "turn_number": spaces.Discrete(1001),
                "phase": spaces.Discrete(3),  # setup, main, finished
                "robber_position": spaces.Box(-2, 2, (3,), dtype=np.int32),
                "bank_resources": spaces.Box(0, 19, (5,), dtype=np.int32),
                "bank_dev_cards": spaces.Discrete(26),
                "longest_road_owner": spaces.Discrete(5),  # 0-3 players, 4=none
                "largest_army_owner": spaces.Discrete(5),
                "player_public_info": spaces.Dict({
                    f"player_{i}": spaces.Dict({
                        "victory_points_visible": spaces.Discrete(15),
                        "num_resources": spaces.Discrete(50),
                        "num_dev_cards": spaces.Discrete(26),
                        "settlements_built": spaces.Discrete(6),
                        "cities_built": spaces.Discrete(5),
                        "roads_built": spaces.Discrete(16),
                        "knights_played": spaces.Discrete(15),
                    }) for i in range(num_players)
                }),
            }),
            # Private state (only for observing agent)
            "private": spaces.Dict({
                "hand_resources": spaces.Box(0, 19, (5,), dtype=np.int32),
                "hand_dev_cards": spaces.Box(0, 5, (5,), dtype=np.int32),
                "actual_victory_points": spaces.Discrete(15),
            }),
            # Action mask
            "action_mask": spaces.MultiBinary(368),
        })

        self.game: Game = None
        self._agent_selector = None

    def reset(self, seed=None, options=None):
        """Reset the environment for a new game."""
        from catanatron.models.map import build_map

        # Create players
        colors = list(self.agent_colors.values())[:self.num_players]
        players = [_ArenaPlayer(color) for color in colors]

        # Initialize Catanatron game
        catan_map = build_map(self.map_type)
        self.game = Game(
            players=players,
            seed=seed,
            catan_map=catan_map,
            vps_to_win=self.vps_to_win,
        )

        # Reset agent tracking
        self.agents = self.possible_agents[:self.num_players]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        # Initialize per-agent state
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.turn_count = 0

        # Sync agent selection with Catanatron's current player
        self._sync_agent_selection()

    def _sync_agent_selection(self):
        """Ensure our agent selection matches Catanatron's current player."""
        if self.game.winning_color() is not None:
            return

        current_color = self.game.state.current_color()
        current_agent = self._color_to_agent(current_color)

        # Advance selector until it matches
        while self.agent_selection != current_agent:
            self.agent_selection = self._agent_selector.next()

    def _color_to_agent(self, color: Color) -> str:
        for agent, c in self.agent_colors.items():
            if c == color:
                return agent
        raise ValueError(f"Unknown color: {color}")

    def _agent_to_color(self, agent: str) -> Color:
        return self.agent_colors[agent]

    def step(self, action: int):
        """Execute an action for the current agent."""
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            # Agent is done, just advance
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        color = self._agent_to_color(agent)

        # Convert action int to Catanatron Action
        playable_actions = self.game.playable_actions
        catan_action = self._action_int_to_catan(action, playable_actions)

        if catan_action is None:
            # Invalid action - apply penalty and skip
            self.rewards[agent] = -0.1
            self.infos[agent]["invalid_action"] = True
        else:
            # Execute valid action
            self.game.execute(catan_action)
            self.rewards[agent] = 0
            self.infos[agent]["action_executed"] = catan_action

        # Check for game end
        winner = self.game.winning_color()
        if winner is not None:
            self._handle_game_end(winner)
        elif self.turn_count >= self.max_turns:
            self._handle_truncation()
        else:
            # Advance to next player
            self._advance_game_state()

        self.turn_count += 1

        # Update agent selection
        self._sync_agent_selection()

    def _advance_game_state(self):
        """Advance Catanatron through any automatic actions (bot moves, etc.)."""
        # In pure Arena mode, all players are external agents
        # No automatic advancement needed
        pass

    def _handle_game_end(self, winner: Color):
        """Set termination flags when game ends."""
        winner_agent = self._color_to_agent(winner)

        for agent in self.agents:
            self.terminations[agent] = True
            if agent == winner_agent:
                self.rewards[agent] = 1.0
            else:
                self.rewards[agent] = -1.0

    def _handle_truncation(self):
        """Handle max turns reached."""
        for agent in self.agents:
            self.truncations[agent] = True
            # Reward based on VP ranking
            vps = get_actual_victory_points(
                self.game.state,
                self._agent_to_color(agent)
            )
            max_vps = max(
                get_actual_victory_points(self.game.state, self._agent_to_color(a))
                for a in self.agents
            )
            self.rewards[agent] = 0.5 if vps == max_vps else -0.5

    def observe(self, agent: str) -> dict:
        """
        Get observation for a specific agent.

        Returns both public state (visible to all) and private state
        (visible only to this agent).
        """
        color = self._agent_to_color(agent)

        return {
            "public": self._get_public_observation(),
            "private": self._get_private_observation(color),
            "action_mask": self._get_action_mask(color),
        }

    def _get_public_observation(self) -> dict:
        """Extract publicly visible game state."""
        state = self.game.state
        board = state.board

        return {
            "board": self._serialize_board(board),
            "turn_number": self.turn_count,
            "phase": self._get_phase(),
            "robber_position": list(board.robber_coordinate),
            "bank_resources": list(state.resource_freqdeck),
            "bank_dev_cards": sum(state.development_deck),
            "longest_road_owner": self._get_longest_road_owner(),
            "largest_army_owner": self._get_largest_army_owner(),
            "player_public_info": {
                agent: self._get_player_public_info(self._agent_to_color(agent))
                for agent in self.agents
            },
        }

    def _get_private_observation(self, color: Color) -> dict:
        """Extract private state visible only to one player."""
        state = self.game.state

        return {
            "hand_resources": get_player_freqdeck(state, color),
            "hand_dev_cards": self._get_dev_cards(state, color),
            "actual_victory_points": get_actual_victory_points(state, color),
        }

    def _get_action_mask(self, color: Color) -> np.ndarray:
        """Generate action mask for valid actions."""
        mask = np.zeros(368, dtype=np.int8)

        # Only mask if it's this player's turn
        if self.game.state.current_color() != color:
            return mask

        for action in self.game.playable_actions:
            action_int = self._catan_action_to_int(action)
            if action_int is not None:
                mask[action_int] = 1

        return mask

    def _serialize_board(self, board) -> dict:
        """Serialize board to JSON-compatible format."""
        # Convert tiles, nodes, edges to serializable format
        tiles = []
        for coord, tile in board.map.tiles.items():
            tiles.append({
                "coordinate": list(coord),
                "type": type(tile).__name__,
                "resource": getattr(tile, "resource", None),
                "number": getattr(tile, "number", None),
            })

        nodes = []
        for node_id, building in board.buildings.items():
            nodes.append({
                "id": node_id,
                "color": building[0].value if building else None,
                "type": building[1] if building else None,
            })

        edges = []
        for edge, color in board.roads.items():
            edges.append({
                "id": list(edge),
                "color": color.value,
            })

        return {"tiles": tiles, "nodes": nodes, "edges": edges}

    def _get_player_public_info(self, color: Color) -> dict:
        """Get publicly visible info about a player."""
        state = self.game.state
        from catanatron.state_functions import (
            get_visible_victory_points,
            player_num_dev_cards,
            get_played_dev_cards,
        )

        return {
            "victory_points_visible": get_visible_victory_points(state, color),
            "num_resources": player_num_resource_cards(state, color),
            "num_dev_cards": player_num_dev_cards(state, color),
            "knights_played": get_played_dev_cards(state, color, "KNIGHT"),
        }

    # Action conversion methods (abbreviated for space)
    def _action_int_to_catan(self, action_int: int, playable_actions) -> Action:
        """Convert integer action to Catanatron Action."""
        # Implementation matches CatanatronEnv.from_action_space
        from catanatron.gym.envs.catanatron_env import ACTIONS_ARRAY, normalize_action

        action_type, value = ACTIONS_ARRAY[action_int]
        for action in playable_actions:
            normalized = normalize_action(action)
            if normalized.action_type == action_type and normalized.value == value:
                return action
        return None

    def _catan_action_to_int(self, action: Action) -> int:
        """Convert Catanatron Action to integer."""
        from catanatron.gym.envs.catanatron_env import to_action_space
        return to_action_space(action)

    # PettingZoo required methods
    def observation_space(self, agent: str) -> spaces.Space:
        return self._observation_space

    def action_space(self, agent: str) -> spaces.Space:
        return self._action_space

    def render(self):
        if self.render_mode == "json":
            return self._get_public_observation()
        return None

    def close(self):
        pass


class _ArenaPlayer(Player):
    """Placeholder player for Arena - all decisions come from external agents."""

    def __init__(self, color: Color):
        super().__init__(color, is_bot=False)

    def decide(self, game, playable_actions):
        # Should never be called - external agents make decisions
        raise RuntimeError("Arena players should not auto-decide")
```

---

## 3. The Data Contract (JSON Schema)

### 3.1 Game Log Schema (Top-Level)

The game log is the **single source of truth** exchanged between the Python backend and React frontend. It must be:

1. **Complete**: Fully replay any game from the log alone
2. **Versioned**: Schema version for backward compatibility
3. **Privacy-aware**: Separate public vs. private state per turn

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://catan-arena.io/schemas/game-log-v1.json",
  "title": "Catan Arena Game Log",
  "type": "object",
  "required": ["schema_version", "game_id", "game_type", "players", "config", "turns", "result"],
  "properties": {
    "schema_version": {
      "const": "1.0.0",
      "description": "Schema version for backward compatibility"
    },
    "game_id": {
      "type": "string",
      "format": "uuid",
      "description": "Unique identifier for this game"
    },
    "game_type": {
      "enum": ["catan", "chess", "diplomacy"],
      "description": "Type of game played"
    },
    "created_at": {
      "type": "string",
      "format": "date-time"
    },
    "duration_seconds": {
      "type": "number"
    },
    "players": {
      "type": "array",
      "items": { "$ref": "#/$defs/Player" },
      "minItems": 2,
      "maxItems": 4
    },
    "config": { "$ref": "#/$defs/GameConfig" },
    "initial_state": { "$ref": "#/$defs/BoardState" },
    "turns": {
      "type": "array",
      "items": { "$ref": "#/$defs/TurnRecord" }
    },
    "result": { "$ref": "#/$defs/GameResult" }
  },
  "$defs": {
    "Player": {
      "type": "object",
      "required": ["id", "color", "agent_type"],
      "properties": {
        "id": { "type": "string" },
        "color": { "enum": ["RED", "BLUE", "ORANGE", "WHITE"] },
        "agent_type": { "enum": ["llm", "human", "random", "minimax"] },
        "model_name": {
          "type": "string",
          "description": "LLM model identifier (e.g., 'claude-3-opus-20240229')"
        },
        "model_config": {
          "type": "object",
          "properties": {
            "temperature": { "type": "number" },
            "max_tokens": { "type": "integer" }
          }
        }
      }
    },
    "GameConfig": {
      "type": "object",
      "properties": {
        "map_type": { "enum": ["BASE", "MINI", "EXTENDED"] },
        "vps_to_win": { "type": "integer", "default": 10 },
        "max_turns": { "type": "integer", "default": 1000 },
        "seed": { "type": "integer" }
      }
    },
    "BoardState": {
      "type": "object",
      "description": "Complete board state at a point in time",
      "properties": {
        "tiles": {
          "type": "array",
          "items": { "$ref": "#/$defs/Tile" }
        },
        "nodes": {
          "type": "array",
          "items": { "$ref": "#/$defs/Node" }
        },
        "edges": {
          "type": "array",
          "items": { "$ref": "#/$defs/Edge" }
        },
        "robber_coordinate": { "$ref": "#/$defs/HexCoordinate" },
        "bank_resources": { "$ref": "#/$defs/ResourceCount" },
        "bank_dev_cards": { "type": "integer" }
      }
    },
    "Tile": {
      "type": "object",
      "required": ["coordinate", "type"],
      "properties": {
        "coordinate": { "$ref": "#/$defs/HexCoordinate" },
        "type": { "enum": ["RESOURCE_TILE", "DESERT", "PORT", "WATER"] },
        "resource": { "$ref": "#/$defs/Resource" },
        "number": { "type": "integer", "minimum": 2, "maximum": 12 },
        "port_type": { "enum": ["THREE_TO_ONE", "WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"] }
      }
    },
    "HexCoordinate": {
      "type": "array",
      "items": { "type": "integer" },
      "minItems": 3,
      "maxItems": 3,
      "description": "Cube coordinates [q, r, s] where q + r + s = 0"
    },
    "Resource": {
      "enum": ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
    },
    "ResourceCount": {
      "type": "object",
      "properties": {
        "WOOD": { "type": "integer" },
        "BRICK": { "type": "integer" },
        "SHEEP": { "type": "integer" },
        "WHEAT": { "type": "integer" },
        "ORE": { "type": "integer" }
      }
    },
    "TurnRecord": {
      "type": "object",
      "required": ["turn_number", "player_id", "phase", "public_state", "action", "timestamp"],
      "properties": {
        "turn_number": { "type": "integer" },
        "player_id": { "type": "string" },
        "phase": { "enum": ["INITIAL_SETTLEMENT", "INITIAL_ROAD", "MAIN", "DISCARD", "MOVE_ROBBER", "TRADE"] },
        "public_state": { "$ref": "#/$defs/PublicTurnState" },
        "private_states": {
          "type": "object",
          "additionalProperties": { "$ref": "#/$defs/PrivateTurnState" },
          "description": "Keyed by player_id. Only current player's state is populated."
        },
        "valid_actions": {
          "type": "array",
          "items": { "$ref": "#/$defs/GameAction" }
        },
        "llm_decision": { "$ref": "#/$defs/LLMDecision" },
        "action": { "$ref": "#/$defs/GameAction" },
        "action_result": { "$ref": "#/$defs/ActionResult" },
        "timestamp": { "type": "string", "format": "date-time" }
      }
    },
    "PublicTurnState": {
      "type": "object",
      "description": "State visible to ALL players",
      "properties": {
        "board": { "$ref": "#/$defs/BoardState" },
        "player_summaries": {
          "type": "object",
          "additionalProperties": {
            "type": "object",
            "properties": {
              "victory_points_visible": { "type": "integer" },
              "resource_count": { "type": "integer", "description": "Total cards, not breakdown" },
              "dev_card_count": { "type": "integer" },
              "settlements_on_board": { "type": "integer" },
              "cities_on_board": { "type": "integer" },
              "roads_on_board": { "type": "integer" },
              "has_longest_road": { "type": "boolean" },
              "has_largest_army": { "type": "boolean" },
              "knights_played": { "type": "integer" }
            }
          }
        },
        "dice_result": {
          "type": "array",
          "items": { "type": "integer" },
          "minItems": 2,
          "maxItems": 2
        }
      }
    },
    "PrivateTurnState": {
      "type": "object",
      "description": "State visible ONLY to one player (their hand)",
      "properties": {
        "hand_resources": { "$ref": "#/$defs/ResourceCount" },
        "hand_dev_cards": {
          "type": "object",
          "properties": {
            "KNIGHT": { "type": "integer" },
            "VICTORY_POINT": { "type": "integer" },
            "ROAD_BUILDING": { "type": "integer" },
            "YEAR_OF_PLENTY": { "type": "integer" },
            "MONOPOLY": { "type": "integer" }
          }
        },
        "actual_victory_points": {
          "type": "integer",
          "description": "Includes hidden VP cards"
        },
        "ports_owned": {
          "type": "array",
          "items": { "enum": ["THREE_TO_ONE", "WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"] }
        }
      }
    },
    "LLMDecision": {
      "type": "object",
      "description": "Complete record of LLM's decision process",
      "properties": {
        "model_name": { "type": "string" },
        "prompt_tokens": { "type": "integer" },
        "completion_tokens": { "type": "integer" },
        "latency_ms": { "type": "integer" },
        "reasoning": {
          "type": "string",
          "description": "Chain-of-Thought text from LLM explaining decision"
        },
        "raw_output": {
          "type": "string",
          "description": "Complete raw response from LLM"
        },
        "parsed_successfully": { "type": "boolean" },
        "parse_error": { "type": "string" }
      }
    },
    "GameAction": {
      "type": "object",
      "required": ["action_type"],
      "properties": {
        "action_type": {
          "enum": [
            "ROLL", "END_TURN", "DISCARD",
            "BUILD_ROAD", "BUILD_SETTLEMENT", "BUILD_CITY",
            "BUY_DEVELOPMENT_CARD",
            "PLAY_KNIGHT_CARD", "PLAY_YEAR_OF_PLENTY", "PLAY_MONOPOLY", "PLAY_ROAD_BUILDING",
            "MOVE_ROBBER",
            "MARITIME_TRADE", "OFFER_TRADE", "ACCEPT_TRADE", "REJECT_TRADE", "CONFIRM_TRADE", "CANCEL_TRADE"
          ]
        },
        "value": {
          "description": "Action-specific parameters"
        }
      }
    },
    "ActionResult": {
      "type": "object",
      "properties": {
        "dice_rolled": {
          "type": "array",
          "items": { "type": "integer" }
        },
        "resources_gained": { "$ref": "#/$defs/ResourceCount" },
        "resources_lost": { "$ref": "#/$defs/ResourceCount" },
        "dev_card_drawn": { "enum": ["KNIGHT", "VICTORY_POINT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY"] },
        "card_stolen": { "$ref": "#/$defs/Resource" }
      }
    },
    "GameResult": {
      "type": "object",
      "required": ["termination_reason"],
      "properties": {
        "termination_reason": { "enum": ["victory", "turn_limit", "player_disconnect", "error"] },
        "winner_id": { "type": "string" },
        "final_scores": {
          "type": "object",
          "additionalProperties": { "type": "integer" }
        },
        "total_turns": { "type": "integer" },
        "statistics": {
          "type": "object",
          "properties": {
            "total_llm_calls": { "type": "integer" },
            "total_tokens_used": { "type": "integer" },
            "average_decision_time_ms": { "type": "number" },
            "invalid_action_attempts": {
              "type": "object",
              "additionalProperties": { "type": "integer" }
            }
          }
        }
      }
    }
  }
}
```

### 3.2 Trade Offer Action Example

Here's how a trade offer action appears in the game log:

```json
{
  "turn_number": 47,
  "player_id": "player_0",
  "phase": "MAIN",
  "public_state": {
    "board": { "...": "..." },
    "player_summaries": {
      "player_0": { "resource_count": 7, "victory_points_visible": 5 },
      "player_1": { "resource_count": 4, "victory_points_visible": 4 },
      "player_2": { "resource_count": 9, "victory_points_visible": 6 },
      "player_3": { "resource_count": 3, "victory_points_visible": 3 }
    },
    "dice_result": [4, 3]
  },
  "private_states": {
    "player_0": {
      "hand_resources": { "WOOD": 2, "BRICK": 1, "SHEEP": 3, "WHEAT": 1, "ORE": 0 },
      "hand_dev_cards": { "KNIGHT": 1, "VICTORY_POINT": 0, "ROAD_BUILDING": 0, "YEAR_OF_PLENTY": 0, "MONOPOLY": 0 },
      "actual_victory_points": 5,
      "ports_owned": ["THREE_TO_ONE", "SHEEP"]
    }
  },
  "valid_actions": [
    { "action_type": "END_TURN" },
    { "action_type": "BUILD_ROAD", "value": [[12, 13]] },
    { "action_type": "MARITIME_TRADE", "value": ["SHEEP", "SHEEP", null, null, "ORE"] },
    { "action_type": "OFFER_TRADE", "value": "any" },
    { "action_type": "PLAY_KNIGHT_CARD" }
  ],
  "llm_decision": {
    "model_name": "claude-3-opus-20240229",
    "prompt_tokens": 2847,
    "completion_tokens": 312,
    "latency_ms": 1823,
    "reasoning": "I have 3 sheep and a sheep port, so I can trade 2:1. However, I notice player_2 has 9 resources and likely has ore (they've been building cities). If I can get ore from them for sheep, that's better than the 2:1 port trade. I'll offer 2 sheep for 1 ore - this is a fair trade since I have the sheep port making sheep less valuable to me, and they likely need sheep for settlements. This trade would let me buy a development card next turn (1 sheep + 1 wheat + 1 ore).",
    "raw_output": "```json\n{\"action_type\": \"OFFER_TRADE\", \"offering\": {\"SHEEP\": 2}, \"requesting\": {\"ORE\": 1}, \"rationale\": \"Leveraging sheep port advantage to acquire ore for dev card\"}\n```",
    "parsed_successfully": true
  },
  "action": {
    "action_type": "OFFER_TRADE",
    "value": {
      "offering": { "WOOD": 0, "BRICK": 0, "SHEEP": 2, "WHEAT": 0, "ORE": 0 },
      "requesting": { "WOOD": 0, "BRICK": 0, "SHEEP": 0, "WHEAT": 0, "ORE": 1 }
    }
  },
  "action_result": {
    "trade_responses": {
      "player_1": "REJECT_TRADE",
      "player_2": "ACCEPT_TRADE",
      "player_3": "REJECT_TRADE"
    }
  },
  "timestamp": "2026-01-24T15:32:47.123Z"
}
```

---

## 4. Agent-Environment Interface

### 4.1 StateAdapter Architecture

The **StateAdapter** is the critical bridge between raw game state and LLM-understandable prompts. It must:

1. Convert complex board geometry into text/ASCII that LLMs can reason about
2. Clearly separate public vs. private information
3. Present valid actions in a parseable format
4. Include enough history for strategic context

```python
# catan_arena/adapters/catan_state_adapter.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass


class StateAdapter(ABC):
    """
    Abstract base class for game-specific state adapters.
    Converts game state to LLM prompts and parses LLM output to actions.
    """

    @abstractmethod
    def state_to_prompt(
        self,
        public_state: dict,
        private_state: dict,
        valid_actions: List[dict],
        turn_history: List[dict],
    ) -> str:
        """Convert game state to an LLM prompt."""
        pass

    @abstractmethod
    def get_output_schema(self) -> dict:
        """Return JSON schema for expected LLM output."""
        pass


@dataclass
class CatanPromptConfig:
    """Configuration for Catan prompt generation."""
    board_representation: str = "ascii"  # "ascii", "list", "both"
    include_probabilities: bool = True
    max_history_turns: int = 10
    include_valid_actions: bool = True
    resource_emoji: bool = False  # Use emoji for resources


class CatanStateAdapter(StateAdapter):
    """
    Converts Catan game state to natural language prompts.

    Two board representation strategies are supported:
    1. ASCII: Visual hex grid (better for spatial reasoning)
    2. List: Structured data (more precise, longer)
    """

    RESOURCE_SYMBOLS = {
        "WOOD": "W",
        "BRICK": "B",
        "SHEEP": "S",
        "WHEAT": "H",  # H for Harvest
        "ORE": "O",
    }

    DICE_PROBABILITIES = {
        2: 1/36, 3: 2/36, 4: 3/36, 5: 4/36, 6: 5/36,
        7: 6/36,
        8: 5/36, 9: 4/36, 10: 3/36, 11: 2/36, 12: 1/36,
    }

    def __init__(self, config: CatanPromptConfig = None):
        self.config = config or CatanPromptConfig()

    def state_to_prompt(
        self,
        public_state: dict,
        private_state: dict,
        valid_actions: List[dict],
        turn_history: List[dict],
    ) -> str:
        """
        Generate a comprehensive prompt for the LLM.

        Structure:
        1. System context (game rules reminder)
        2. Current board state
        3. Your hand (private)
        4. Other players' visible state
        5. Recent history
        6. Available actions
        7. Output format instructions
        """
        sections = [
            self._system_context(),
            self._board_section(public_state),
            self._your_state_section(private_state),
            self._opponents_section(public_state),
            self._history_section(turn_history),
            self._actions_section(valid_actions),
            self._output_instructions(),
        ]

        return "\n\n".join(sections)

    def _system_context(self) -> str:
        return """# SETTLERS OF CATAN - AI PLAYER

You are playing Settlers of Catan. Your goal is to reach 10 Victory Points first.

## Victory Point Sources:
- Settlement: 1 VP
- City: 2 VP
- Longest Road (5+ roads): 2 VP
- Largest Army (3+ knights): 2 VP
- Victory Point cards: 1 VP each (hidden until winning)

## Building Costs:
- Road: 1 Wood + 1 Brick
- Settlement: 1 Wood + 1 Brick + 1 Sheep + 1 Wheat
- City (upgrade): 2 Wheat + 3 Ore
- Development Card: 1 Sheep + 1 Wheat + 1 Ore

## Key Rules:
- You must roll dice at the start of your turn
- You can only build on your turn after rolling
- Settlements must be 2+ roads apart
- 7 rolled = robber moves, players with 8+ cards discard half
- You can trade with the bank (4:1) or at ports (3:1 or 2:1)"""

    def _board_section(self, public_state: dict) -> str:
        """Generate the board representation."""
        board = public_state.get("board", {})

        if self.config.board_representation == "ascii":
            return self._ascii_board(board)
        elif self.config.board_representation == "list":
            return self._list_board(board)
        else:  # "both"
            return self._ascii_board(board) + "\n\n" + self._list_board(board)

    def _ascii_board(self, board: dict) -> str:
        """
        Generate ASCII representation of the hex board.

        Standard Catan board layout (cube coordinates):

                  [0,-2,2]  [1,-2,1]  [2,-2,0]
              [-1,-1,2] [0,-1,1] [1,-1,0] [2,-1,-1]
          [-2,0,2] [-1,0,1] [0,0,0] [1,0,-1] [2,0,-2]
              [-2,1,1] [-1,1,0] [0,1,-1] [1,1,-2]
                  [-2,2,0]  [-1,2,-1]  [0,2,-2]
        """
        tiles = board.get("tiles", [])

        # Build coordinate to tile mapping
        tile_map = {}
        for tile in tiles:
            coord = tuple(tile["coordinate"])
            tile_map[coord] = tile

        # ASCII art generation
        lines = ["## BOARD STATE (ASCII)"]
        lines.append("```")
        lines.append("Legend: W=Wood B=Brick S=Sheep H=Wheat O=Ore D=Desert *=Robber")
        lines.append("        Numbers show dice roll. Format: [Resource:Number]")
        lines.append("")

        # Row by row rendering (simplified)
        rows = [
            [(0,-2,2), (1,-2,1), (2,-2,0)],
            [(-1,-1,2), (0,-1,1), (1,-1,0), (2,-1,-1)],
            [(-2,0,2), (-1,0,1), (0,0,0), (1,0,-1), (2,0,-2)],
            [(-2,1,1), (-1,1,0), (0,1,-1), (1,1,-2)],
            [(-2,2,0), (-1,2,-1), (0,2,-2)],
        ]

        robber_coord = tuple(board.get("robber_coordinate", [0,0,0]))

        for row_idx, row in enumerate(rows):
            indent = "  " * (2 - abs(row_idx - 2))
            row_str = indent
            for coord in row:
                tile = tile_map.get(coord)
                if tile:
                    cell = self._tile_to_ascii(tile, coord == robber_coord)
                else:
                    cell = "[    ]"
                row_str += cell + " "
            lines.append(row_str)

        lines.append("```")

        # Add buildings
        lines.append("\n### BUILDINGS ON BOARD")
        buildings = self._format_buildings(board)
        lines.append(buildings)

        return "\n".join(lines)

    def _tile_to_ascii(self, tile: dict, has_robber: bool) -> str:
        """Convert a single tile to ASCII representation."""
        tile_type = tile.get("type")

        if tile_type == "DESERT":
            return "[D:--]" + ("*" if has_robber else " ")
        elif tile_type == "RESOURCE_TILE":
            resource = tile.get("resource", "?")
            number = tile.get("number", 0)
            symbol = self.RESOURCE_SYMBOLS.get(resource, "?")
            robber = "*" if has_robber else ""
            return f"[{symbol}:{number:2d}]{robber}"
        elif tile_type == "PORT":
            port_resource = tile.get("port_type", "3:1")
            return f"[P:{port_resource[:2]}]"
        else:
            return "[    ]"

    def _list_board(self, board: dict) -> str:
        """Generate structured list representation of the board."""
        lines = ["## BOARD STATE (Structured)"]

        # Resource tiles
        lines.append("\n### Resource Tiles:")
        for tile in board.get("tiles", []):
            if tile.get("type") == "RESOURCE_TILE":
                coord = tile["coordinate"]
                resource = tile["resource"]
                number = tile["number"]
                prob = self.DICE_PROBABILITIES.get(number, 0)
                lines.append(f"  - Tile {coord}: {resource} with number {number} (probability: {prob:.1%})")

        # Ports
        lines.append("\n### Ports:")
        for tile in board.get("tiles", []):
            if tile.get("type") == "PORT":
                coord = tile["coordinate"]
                port_type = tile.get("port_type", "THREE_TO_ONE")
                lines.append(f"  - Port at {coord}: {port_type}")

        return "\n".join(lines)

    def _format_buildings(self, board: dict) -> str:
        """Format buildings (settlements, cities, roads) by player."""
        lines = []

        # Group by color
        by_color = {"RED": [], "BLUE": [], "ORANGE": [], "WHITE": []}

        for node in board.get("nodes", []):
            if node.get("color"):
                by_color[node["color"]].append(
                    f"  - {node['type']} at node {node['id']}"
                )

        for edge in board.get("edges", []):
            if edge.get("color"):
                by_color[edge["color"]].append(
                    f"  - Road at edge {edge['id']}"
                )

        for color, buildings in by_color.items():
            if buildings:
                lines.append(f"\n{color}:")
                lines.extend(buildings)

        return "\n".join(lines) if lines else "No buildings yet."

    def _your_state_section(self, private_state: dict) -> str:
        """Format the player's private state (their hand)."""
        lines = ["## YOUR STATE (Private - only you can see this)"]

        # Resources
        hand = private_state.get("hand_resources", {})
        total = sum(hand.values())
        lines.append(f"\n### Your Hand ({total} cards):")
        for resource, count in hand.items():
            if count > 0:
                lines.append(f"  - {resource}: {count}")

        # Dev cards
        dev_cards = private_state.get("hand_dev_cards", {})
        total_dev = sum(dev_cards.values())
        if total_dev > 0:
            lines.append(f"\n### Your Development Cards ({total_dev}):")
            for card, count in dev_cards.items():
                if count > 0:
                    lines.append(f"  - {card}: {count}")

        # Victory points
        vp = private_state.get("actual_victory_points", 0)
        lines.append(f"\n### Your Victory Points: {vp}")

        # Ports
        ports = private_state.get("ports_owned", [])
        if ports:
            lines.append(f"\n### Your Ports: {', '.join(ports)}")

        return "\n".join(lines)

    def _opponents_section(self, public_state: dict) -> str:
        """Format publicly visible information about opponents."""
        lines = ["## OTHER PLAYERS (Public Information)"]

        summaries = public_state.get("player_summaries", {})
        for player_id, info in summaries.items():
            lines.append(f"\n### {player_id}:")
            lines.append(f"  - Visible VP: {info.get('victory_points_visible', 0)}")
            lines.append(f"  - Resource cards: {info.get('resource_count', 0)}")
            lines.append(f"  - Dev cards: {info.get('dev_card_count', 0)}")
            lines.append(f"  - Knights played: {info.get('knights_played', 0)}")

            if info.get("has_longest_road"):
                lines.append("  - ** HAS LONGEST ROAD **")
            if info.get("has_largest_army"):
                lines.append("  - ** HAS LARGEST ARMY **")

        return "\n".join(lines)

    def _history_section(self, turn_history: List[dict]) -> str:
        """Format recent game history."""
        if not turn_history:
            return "## RECENT HISTORY\nNo previous turns."

        lines = ["## RECENT HISTORY (Last 10 turns)"]

        for turn in turn_history[-10:]:
            player = turn.get("current_player", "?")
            action = turn.get("action", {})
            action_type = action.get("action_type", "?")

            # Format based on action type
            if action_type == "ROLL":
                result = turn.get("action_result", {}).get("dice_rolled", [])
                lines.append(f"  Turn {turn['turn_number']}: {player} rolled {sum(result)} ({result})")
            elif action_type in ["BUILD_SETTLEMENT", "BUILD_CITY", "BUILD_ROAD"]:
                lines.append(f"  Turn {turn['turn_number']}: {player} {action_type.replace('_', ' ').lower()}")
            elif action_type == "OFFER_TRADE":
                value = action.get("value", {})
                lines.append(f"  Turn {turn['turn_number']}: {player} offered trade")
            else:
                lines.append(f"  Turn {turn['turn_number']}: {player} {action_type}")

        return "\n".join(lines)

    def _actions_section(self, valid_actions: List[dict]) -> str:
        """Format available actions."""
        lines = ["## AVAILABLE ACTIONS"]
        lines.append("You must choose ONE of these actions:")
        lines.append("")

        # Group by action type
        by_type = {}
        for action in valid_actions:
            atype = action.get("action_type")
            if atype not in by_type:
                by_type[atype] = []
            by_type[atype].append(action)

        for atype, actions in by_type.items():
            if len(actions) == 1 and actions[0].get("value") is None:
                lines.append(f"- {atype}")
            else:
                lines.append(f"- {atype}:")
                for action in actions[:5]:  # Limit to 5 examples
                    value = action.get("value")
                    if value:
                        lines.append(f"    - {value}")
                if len(actions) > 5:
                    lines.append(f"    - ... and {len(actions) - 5} more options")

        return "\n".join(lines)

    def _output_instructions(self) -> str:
        """Instructions for LLM output format."""
        return """## YOUR RESPONSE

Think through your decision step by step, then output your chosen action.

Your response MUST end with a JSON code block in this exact format:

```json
{
  "action_type": "BUILD_SETTLEMENT",
  "value": 23,
  "rationale": "Brief explanation of why this action"
}
```

Valid action_type values: ROLL, END_TURN, BUILD_ROAD, BUILD_SETTLEMENT, BUILD_CITY,
BUY_DEVELOPMENT_CARD, PLAY_KNIGHT_CARD, PLAY_YEAR_OF_PLENTY, PLAY_MONOPOLY,
PLAY_ROAD_BUILDING, MOVE_ROBBER, MARITIME_TRADE, OFFER_TRADE, ACCEPT_TRADE,
REJECT_TRADE, DISCARD

The "value" field depends on action_type:
- BUILD_ROAD: [node1, node2] edge
- BUILD_SETTLEMENT/CITY: node_id (integer)
- MOVE_ROBBER: {"coordinate": [q,r,s], "steal_from": "player_id" or null}
- MARITIME_TRADE: {"give": {"RESOURCE": count}, "receive": {"RESOURCE": count}}
- OFFER_TRADE: {"offering": {"RESOURCE": count}, "requesting": {"RESOURCE": count}}
- PLAY_YEAR_OF_PLENTY: [resource1, resource2]
- PLAY_MONOPOLY: "RESOURCE"
- For ROLL, END_TURN, BUY_DEVELOPMENT_CARD, etc.: null"""

    def get_output_schema(self) -> dict:
        """JSON Schema for expected LLM output."""
        return {
            "type": "object",
            "required": ["action_type", "rationale"],
            "properties": {
                "action_type": {
                    "type": "string",
                    "enum": [
                        "ROLL", "END_TURN", "DISCARD",
                        "BUILD_ROAD", "BUILD_SETTLEMENT", "BUILD_CITY",
                        "BUY_DEVELOPMENT_CARD",
                        "PLAY_KNIGHT_CARD", "PLAY_YEAR_OF_PLENTY",
                        "PLAY_MONOPOLY", "PLAY_ROAD_BUILDING",
                        "MOVE_ROBBER",
                        "MARITIME_TRADE", "OFFER_TRADE",
                        "ACCEPT_TRADE", "REJECT_TRADE"
                    ]
                },
                "value": {
                    "description": "Action-specific parameters (varies by action_type)"
                },
                "rationale": {
                    "type": "string",
                    "description": "Brief explanation of decision"
                }
            }
        }
```

### 4.2 ActionParser

```python
# catan_arena/adapters/action_parser.py

import json
import re
from typing import List, Dict, Any, Optional


class ActionParseError(Exception):
    """Raised when LLM output cannot be parsed into a valid action."""
    pass


class CatanActionParser:
    """
    Parses LLM output into Catan game actions.

    Supports multiple output formats:
    1. Strict JSON in code block
    2. JSON without code block
    3. Natural language fallback (experimental)
    """

    def parse(
        self,
        raw_output: str,
        valid_actions: List[dict]
    ) -> dict:
        """
        Parse LLM output into a valid game action.

        Args:
            raw_output: Complete LLM response text
            valid_actions: List of valid actions for this turn

        Returns:
            Parsed action dict matching one of valid_actions

        Raises:
            ActionParseError: If output cannot be parsed or action is invalid
        """
        # Try JSON extraction first
        parsed = self._extract_json(raw_output)

        if parsed is None:
            # Fallback to natural language parsing
            parsed = self._parse_natural_language(raw_output, valid_actions)

        if parsed is None:
            raise ActionParseError(
                f"Could not extract action from output: {raw_output[:200]}..."
            )

        # Validate against valid_actions
        validated = self._validate_action(parsed, valid_actions)

        if validated is None:
            raise ActionParseError(
                f"Action {parsed} is not in valid_actions list"
            )

        return validated

    def _extract_json(self, text: str) -> Optional[dict]:
        """Extract JSON from text, handling code blocks."""
        # Try to find JSON code block
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, text, re.DOTALL)

        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find raw JSON object
        json_obj_pattern = r'\{[^{}]*"action_type"[^{}]*\}'
        match = re.search(json_obj_pattern, text, re.DOTALL)

        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def _parse_natural_language(
        self,
        text: str,
        valid_actions: List[dict]
    ) -> Optional[dict]:
        """
        Attempt to parse natural language into an action.
        This is a fallback when JSON parsing fails.
        """
        text_lower = text.lower()

        # Simple keyword matching
        if "roll" in text_lower and "dice" in text_lower:
            return {"action_type": "ROLL", "value": None}

        if "end" in text_lower and "turn" in text_lower:
            return {"action_type": "END_TURN", "value": None}

        if "build" in text_lower and "road" in text_lower:
            # Try to extract edge
            edge_match = re.search(r'\[(\d+),\s*(\d+)\]', text)
            if edge_match:
                return {
                    "action_type": "BUILD_ROAD",
                    "value": [int(edge_match.group(1)), int(edge_match.group(2))]
                }

        if "build" in text_lower and "settlement" in text_lower:
            # Try to extract node
            node_match = re.search(r'node\s*(\d+)', text_lower)
            if node_match:
                return {
                    "action_type": "BUILD_SETTLEMENT",
                    "value": int(node_match.group(1))
                }

        # Could not parse
        return None

    def _validate_action(
        self,
        parsed: dict,
        valid_actions: List[dict]
    ) -> Optional[dict]:
        """
        Validate parsed action against list of valid actions.
        Returns the matching valid action or None.
        """
        action_type = parsed.get("action_type")
        value = parsed.get("value")

        for valid in valid_actions:
            if valid["action_type"] != action_type:
                continue

            # For actions without value, just match type
            if valid.get("value") is None and value is None:
                return valid

            # For actions with value, check if value matches
            if self._values_match(value, valid.get("value")):
                return valid

        # Try fuzzy matching for value
        for valid in valid_actions:
            if valid["action_type"] != action_type:
                continue

            if self._values_fuzzy_match(value, valid.get("value")):
                return valid

        return None

    def _values_match(self, parsed_value, valid_value) -> bool:
        """Check if parsed value matches valid action value."""
        if parsed_value == valid_value:
            return True

        # Handle list/tuple equivalence
        if isinstance(parsed_value, list) and isinstance(valid_value, (list, tuple)):
            return list(parsed_value) == list(valid_value)

        return False

    def _values_fuzzy_match(self, parsed_value, valid_value) -> bool:
        """Attempt fuzzy matching for edge cases."""
        # Handle sorted edge tuples
        if isinstance(parsed_value, list) and len(parsed_value) == 2:
            sorted_parsed = tuple(sorted(parsed_value))
            if isinstance(valid_value, (list, tuple)) and len(valid_value) == 2:
                sorted_valid = tuple(sorted(valid_value))
                return sorted_parsed == sorted_valid

        return False
```

---

## 5. The React Replay Viewer

### 5.1 Architecture Overview

The React Replay Viewer loads JSON game logs and provides:

1. **Turn-by-turn playback** with play/pause/seek
2. **Fog of War toggle**: See what each LLM saw
3. **LLM reasoning display**: Show Chain-of-Thought for each decision
4. **Timeline visualization**: Game progression and key events

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           REPLAY VIEWER ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        GameLogProvider (Context)                     │   │
│  │  - Loads JSON log file                                               │   │
│  │  - Manages current turn index                                        │   │
│  │  - Provides turn data to children                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│           ┌────────────────────────┼────────────────────────┐              │
│           │                        │                        │              │
│           ▼                        ▼                        ▼              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │  BoardRenderer  │    │  PlaybackControl │    │  InfoPanels     │        │
│  │                 │    │                 │    │                 │        │
│  │  - Hex grid     │    │  - Play/Pause   │    │  - Player hands │        │
│  │  - Buildings    │    │  - Speed        │    │  - LLM reasoning│        │
│  │  - Robber       │    │  - Seek slider  │    │  - Turn history │        │
│  │  - Animations   │    │  - Turn counter │    │  - Statistics   │        │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘        │
│           │                                              │                  │
│           │              ┌─────────────────┐            │                  │
│           └──────────────│   ViewMode      │────────────┘                  │
│                          │   Controller    │                               │
│                          │                 │                               │
│                          │  - Omniscient   │                               │
│                          │  - Player POV   │                               │
│                          │  - Spectator    │                               │
│                          └─────────────────┘                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Core Components

```tsx
// src/components/replay/GameLogProvider.tsx

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';

interface GameLog {
  schema_version: string;
  game_id: string;
  game_type: string;
  players: Player[];
  turns: TurnRecord[];
  result: GameResult;
}

interface TurnRecord {
  turn_number: number;
  player_id: string;
  public_state: PublicTurnState;
  private_states: Record<string, PrivateTurnState>;
  llm_decision?: LLMDecision;
  action: GameAction;
}

interface ViewMode {
  type: 'omniscient' | 'player_pov' | 'spectator';
  player_id?: string;  // For player_pov mode
}

interface GameLogContextValue {
  // Data
  gameLog: GameLog | null;
  currentTurn: number;
  totalTurns: number;
  currentTurnData: TurnRecord | null;

  // Playback controls
  isPlaying: boolean;
  playbackSpeed: number;
  play: () => void;
  pause: () => void;
  seekToTurn: (turn: number) => void;
  nextTurn: () => void;
  prevTurn: () => void;
  setPlaybackSpeed: (speed: number) => void;

  // View mode (Fog of War)
  viewMode: ViewMode;
  setViewMode: (mode: ViewMode) => void;

  // Computed state based on view mode
  visibleState: VisibleGameState;
}

const GameLogContext = createContext<GameLogContextValue | null>(null);

export function GameLogProvider({
  children,
  logUrl
}: {
  children: React.ReactNode;
  logUrl: string;
}) {
  const [gameLog, setGameLog] = useState<GameLog | null>(null);
  const [currentTurn, setCurrentTurn] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [viewMode, setViewMode] = useState<ViewMode>({ type: 'omniscient' });

  // Load game log
  useEffect(() => {
    fetch(logUrl)
      .then(res => res.json())
      .then(data => {
        setGameLog(data);
        setCurrentTurn(0);
      });
  }, [logUrl]);

  // Playback timer
  useEffect(() => {
    if (!isPlaying || !gameLog) return;

    const interval = setInterval(() => {
      setCurrentTurn(prev => {
        if (prev >= gameLog.turns.length - 1) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 1000 / playbackSpeed);

    return () => clearInterval(interval);
  }, [isPlaying, playbackSpeed, gameLog]);

  // Compute visible state based on view mode
  const visibleState = useMemo(() => {
    if (!gameLog || currentTurn >= gameLog.turns.length) {
      return null;
    }

    const turnData = gameLog.turns[currentTurn];
    return computeVisibleState(turnData, viewMode, gameLog.players);
  }, [gameLog, currentTurn, viewMode]);

  const value: GameLogContextValue = {
    gameLog,
    currentTurn,
    totalTurns: gameLog?.turns.length ?? 0,
    currentTurnData: gameLog?.turns[currentTurn] ?? null,
    isPlaying,
    playbackSpeed,
    play: () => setIsPlaying(true),
    pause: () => setIsPlaying(false),
    seekToTurn: setCurrentTurn,
    nextTurn: () => setCurrentTurn(t => Math.min(t + 1, (gameLog?.turns.length ?? 1) - 1)),
    prevTurn: () => setCurrentTurn(t => Math.max(t - 1, 0)),
    setPlaybackSpeed,
    viewMode,
    setViewMode,
    visibleState,
  };

  return (
    <GameLogContext.Provider value={value}>
      {children}
    </GameLogContext.Provider>
  );
}

// Helper function to compute what's visible based on view mode
function computeVisibleState(
  turnData: TurnRecord,
  viewMode: ViewMode,
  players: Player[]
): VisibleGameState {
  const { public_state, private_states } = turnData;

  switch (viewMode.type) {
    case 'omniscient':
      // Show everything - all players' hands visible
      return {
        board: public_state.board,
        playerStates: Object.fromEntries(
          players.map(p => [
            p.id,
            {
              ...public_state.player_summaries[p.id],
              ...private_states[p.id],  // Include private state
              handVisible: true,
            }
          ])
        ),
        showAllReasoning: true,
      };

    case 'player_pov':
      // Show only what the selected player could see at this turn
      const povPlayerId = viewMode.player_id!;
      return {
        board: public_state.board,
        playerStates: Object.fromEntries(
          players.map(p => {
            if (p.id === povPlayerId) {
              // This player sees their own hand
              return [p.id, {
                ...public_state.player_summaries[p.id],
                ...private_states[p.id],
                handVisible: true,
              }];
            } else {
              // Other players' hands are hidden
              return [p.id, {
                ...public_state.player_summaries[p.id],
                handVisible: false,
              }];
            }
          })
        ),
        showAllReasoning: false,
        showReasoningFor: povPlayerId,
      };

    case 'spectator':
      // Show only public information (like watching a live game)
      return {
        board: public_state.board,
        playerStates: Object.fromEntries(
          players.map(p => [
            p.id,
            {
              ...public_state.player_summaries[p.id],
              handVisible: false,
            }
          ])
        ),
        showAllReasoning: false,
      };
  }
}
```

### 5.3 Fog of War Controller

```tsx
// src/components/replay/ViewModeController.tsx

import React from 'react';
import {
  ToggleButtonGroup,
  ToggleButton,
  Select,
  MenuItem,
  Box,
  Typography,
  Tooltip
} from '@mui/material';
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
import PersonIcon from '@mui/icons-material/Person';

interface ViewModeControllerProps {
  viewMode: ViewMode;
  onViewModeChange: (mode: ViewMode) => void;
  players: Player[];
}

export function ViewModeController({
  viewMode,
  onViewModeChange,
  players
}: ViewModeControllerProps) {
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
      <Typography variant="subtitle2">View Mode:</Typography>

      <ToggleButtonGroup
        value={viewMode.type}
        exclusive
        onChange={(_, value) => {
          if (value === 'player_pov') {
            onViewModeChange({ type: 'player_pov', player_id: players[0].id });
          } else if (value) {
            onViewModeChange({ type: value });
          }
        }}
        size="small"
      >
        <Tooltip title="Omniscient - See all players' hands and reasoning">
          <ToggleButton value="omniscient">
            <VisibilityIcon sx={{ mr: 1 }} />
            Omniscient
          </ToggleButton>
        </Tooltip>

        <Tooltip title="Player POV - See only what one player saw">
          <ToggleButton value="player_pov">
            <PersonIcon sx={{ mr: 1 }} />
            Player View
          </ToggleButton>
        </Tooltip>

        <Tooltip title="Spectator - Public information only">
          <ToggleButton value="spectator">
            <VisibilityOffIcon sx={{ mr: 1 }} />
            Spectator
          </ToggleButton>
        </Tooltip>
      </ToggleButtonGroup>

      {viewMode.type === 'player_pov' && (
        <Select
          value={viewMode.player_id}
          onChange={(e) => onViewModeChange({
            type: 'player_pov',
            player_id: e.target.value
          })}
          size="small"
        >
          {players.map(player => (
            <MenuItem key={player.id} value={player.id}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box
                  sx={{
                    width: 12,
                    height: 12,
                    borderRadius: '50%',
                    backgroundColor: player.color.toLowerCase(),
                  }}
                />
                {player.model_name || player.id}
              </Box>
            </MenuItem>
          ))}
        </Select>
      )}
    </Box>
  );
}
```

### 5.4 LLM Reasoning Panel

```tsx
// src/components/replay/LLMReasoningPanel.tsx

import React from 'react';
import {
  Paper,
  Typography,
  Box,
  Chip,
  Collapse,
  IconButton,
  Divider
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import TimerIcon from '@mui/icons-material/Timer';
import TokenIcon from '@mui/icons-material/DataObject';

interface LLMReasoningPanelProps {
  decision: LLMDecision | undefined;
  action: GameAction;
  isVisible: boolean;  // Based on view mode
}

export function LLMReasoningPanel({
  decision,
  action,
  isVisible
}: LLMReasoningPanelProps) {
  const [expanded, setExpanded] = React.useState(true);

  if (!decision || !isVisible) {
    return (
      <Paper sx={{ p: 2, opacity: 0.5 }}>
        <Typography variant="body2" color="text.secondary">
          LLM reasoning hidden in this view mode
        </Typography>
      </Paper>
    );
  }

  return (
    <Paper sx={{ p: 2 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <SmartToyIcon color="primary" />
          <Typography variant="h6">{decision.model_name}</Typography>
        </Box>

        <Box sx={{ display: 'flex', gap: 1 }}>
          <Chip
            icon={<TimerIcon />}
            label={`${decision.latency_ms}ms`}
            size="small"
            variant="outlined"
          />
          <Chip
            icon={<TokenIcon />}
            label={`${decision.prompt_tokens + decision.completion_tokens} tokens`}
            size="small"
            variant="outlined"
          />
        </Box>
      </Box>

      {/* Action Taken */}
      <Box sx={{ mb: 2 }}>
        <Typography variant="subtitle2" color="text.secondary">
          Action Taken:
        </Typography>
        <Chip
          label={action.action_type}
          color="primary"
          sx={{ mr: 1 }}
        />
        {action.value && (
          <Typography variant="body2" component="span">
            {JSON.stringify(action.value)}
          </Typography>
        )}
      </Box>

      <Divider sx={{ my: 2 }} />

      {/* Chain of Thought */}
      <Box>
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            cursor: 'pointer',
          }}
          onClick={() => setExpanded(!expanded)}
        >
          <Typography variant="subtitle2" color="text.secondary">
            Chain of Thought:
          </Typography>
          <IconButton size="small">
            <ExpandMoreIcon
              sx={{
                transform: expanded ? 'rotate(180deg)' : 'none',
                transition: 'transform 0.2s',
              }}
            />
          </IconButton>
        </Box>

        <Collapse in={expanded}>
          <Paper
            variant="outlined"
            sx={{
              p: 2,
              mt: 1,
              backgroundColor: 'grey.50',
              fontFamily: 'monospace',
              fontSize: '0.875rem',
              whiteSpace: 'pre-wrap',
            }}
          >
            {decision.reasoning}
          </Paper>
        </Collapse>
      </Box>

      {/* Parse Status */}
      {!decision.parsed_successfully && (
        <Box sx={{ mt: 2 }}>
          <Chip
            label="Parse Error"
            color="error"
            size="small"
          />
          <Typography variant="body2" color="error" sx={{ mt: 1 }}>
            {decision.parse_error}
          </Typography>
        </Box>
      )}
    </Paper>
  );
}
```

### 5.5 Main Replay Page

```tsx
// src/pages/ReplayPage.tsx

import React from 'react';
import { useParams } from 'react-router-dom';
import { Box, Grid, Paper } from '@mui/material';
import { GameLogProvider, useGameLog } from '../components/replay/GameLogProvider';
import { BoardRenderer } from '../components/replay/BoardRenderer';
import { PlaybackControls } from '../components/replay/PlaybackControls';
import { ViewModeController } from '../components/replay/ViewModeController';
import { LLMReasoningPanel } from '../components/replay/LLMReasoningPanel';
import { PlayerHandsPanel } from '../components/replay/PlayerHandsPanel';
import { TurnTimeline } from '../components/replay/TurnTimeline';

function ReplayContent() {
  const {
    gameLog,
    currentTurnData,
    visibleState,
    viewMode,
    setViewMode,
  } = useGameLog();

  if (!gameLog || !visibleState) {
    return <div>Loading...</div>;
  }

  return (
    <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Top Bar - View Mode Controls */}
      <Paper sx={{ p: 2 }}>
        <ViewModeController
          viewMode={viewMode}
          onViewModeChange={setViewMode}
          players={gameLog.players}
        />
      </Paper>

      {/* Main Content */}
      <Grid container sx={{ flex: 1, overflow: 'hidden' }}>
        {/* Left Panel - Player Hands */}
        <Grid item xs={2}>
          <PlayerHandsPanel
            players={gameLog.players}
            playerStates={visibleState.playerStates}
          />
        </Grid>

        {/* Center - Board */}
        <Grid item xs={7}>
          <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Box sx={{ flex: 1 }}>
              <BoardRenderer
                board={visibleState.board}
                highlightAction={currentTurnData?.action}
              />
            </Box>

            {/* Playback Controls */}
            <PlaybackControls />
          </Box>
        </Grid>

        {/* Right Panel - LLM Reasoning */}
        <Grid item xs={3}>
          <LLMReasoningPanel
            decision={currentTurnData?.llm_decision}
            action={currentTurnData?.action}
            isVisible={
              visibleState.showAllReasoning ||
              visibleState.showReasoningFor === currentTurnData?.player_id
            }
          />
        </Grid>
      </Grid>

      {/* Bottom - Timeline */}
      <TurnTimeline />
    </Box>
  );
}

export default function ReplayPage() {
  const { gameId } = useParams<{ gameId: string }>();
  const logUrl = `/api/logs/${gameId}.json`;

  return (
    <GameLogProvider logUrl={logUrl}>
      <ReplayContent />
    </GameLogProvider>
  );
}
```

---

## 6. Extensibility Plan

### 6.1 Game-Agnostic Architecture

The Catan-Arena architecture is designed so that adding a new game requires implementing only:

1. **PettingZoo Environment Wrapper** - Wraps the game engine
2. **StateAdapter** - Converts game state to LLM prompts
3. **ActionParser** - Parses LLM output to game actions
4. **BoardRenderer** (React) - Visualizes the game board

The following components remain unchanged:

| Component | Reusable? | Notes |
|-----------|-----------|-------|
| LangGraph Orchestration | **100%** | Game loop is generic |
| Game Log Schema | **90%** | Base schema + game-specific extensions |
| Replay Viewer (React) | **80%** | Timeline, playback, reasoning panels reusable |
| LLM Integration | **100%** | LangChain handles all models |
| Checkpointing | **100%** | LangGraph's built-in persistence |

### 6.2 Example: Adding Chess

```python
# catan_arena/games/chess/__init__.py

"""
Chess game integration for Catan-Arena.
Uses python-chess as the engine.
"""

# 1. PettingZoo Environment
from .env import ChessAECEnv

# 2. State Adapter
from .adapter import ChessStateAdapter

# 3. Action Parser
from .parser import ChessActionParser

# Registry entry
GAME_REGISTRY = {
    "chess": {
        "env_class": ChessAECEnv,
        "adapter_class": ChessStateAdapter,
        "parser_class": ChessActionParser,
    }
}
```

```python
# catan_arena/games/chess/adapter.py

from catan_arena.adapters.base import StateAdapter

class ChessStateAdapter(StateAdapter):
    """
    Converts chess position to LLM prompt.

    Uses:
    - ASCII board representation
    - FEN notation for precise state
    - Move history in algebraic notation
    """

    def state_to_prompt(
        self,
        public_state: dict,
        private_state: dict,  # Chess has no private state
        valid_actions: List[dict],
        turn_history: List[dict],
    ) -> str:
        board_ascii = self._render_ascii_board(public_state["board"])
        fen = public_state["fen"]

        return f"""# CHESS GAME

## Current Position
```
{board_ascii}
```

FEN: {fen}

## Your Color: {public_state["to_move"]}

## Legal Moves:
{self._format_moves(valid_actions)}

## Game History (last 10 moves):
{self._format_history(turn_history)}

## Your Response

Analyze the position, then output your move in this format:

```json
{{"move": "e2e4", "rationale": "Opening with King's Pawn"}}
```
"""

    def _render_ascii_board(self, board) -> str:
        """Render chess board as ASCII."""
        lines = ["  a b c d e f g h"]
        for rank in range(8, 0, -1):
            row = f"{rank} "
            for file in "abcdefgh":
                piece = board.get(f"{file}{rank}", ".")
                row += piece + " "
            lines.append(row)
        return "\n".join(lines)

    def get_output_schema(self) -> dict:
        return {
            "type": "object",
            "required": ["move"],
            "properties": {
                "move": {
                    "type": "string",
                    "pattern": "^[a-h][1-8][a-h][1-8][qrbn]?$",
                    "description": "Move in UCI format (e.g., 'e2e4', 'e7e8q')"
                },
                "rationale": {"type": "string"}
            }
        }
```

### 6.3 Component Replacement Matrix

| When Adding New Game | Replace | Keep |
|---------------------|---------|------|
| **Backend** | | |
| Game Engine | New engine library | - |
| PettingZoo Wrapper | ✓ Implement | - |
| StateAdapter | ✓ Implement | Base class |
| ActionParser | ✓ Implement | Base class |
| LangGraph Nodes | - | ✓ All nodes |
| LLM Integration | - | ✓ LangChain setup |
| **Frontend** | | |
| Board Renderer | ✓ Implement | - |
| Replay Timeline | - | ✓ Reuse |
| Playback Controls | - | ✓ Reuse |
| LLM Reasoning Panel | - | ✓ Reuse |
| View Mode Controller | Possibly extend | ✓ Base logic |
| **Data** | | |
| Game Log Schema | Extend `$defs` | ✓ Core schema |
| Checkpoints | - | ✓ LangGraph built-in |

---

## 7. Implementation Roadmap

### Phase 1: Core Infrastructure (Weeks 1-2)

- [ ] Implement `CatanAECEnv` PettingZoo wrapper
- [ ] Implement `CatanStateAdapter` with ASCII board
- [ ] Implement `CatanActionParser`
- [ ] Build LangGraph workflow with all nodes
- [ ] Add checkpointing with SQLite

### Phase 2: LLM Integration (Weeks 3-4)

- [ ] Integrate Claude via LangChain
- [ ] Integrate GPT-4 via LangChain
- [ ] Integrate Gemini via LangChain
- [ ] Implement retry logic and error handling
- [ ] Add token usage tracking

### Phase 3: Game Log & Storage (Week 5)

- [ ] Finalize JSON schema
- [ ] Implement game log writer
- [ ] Add compression for large logs
- [ ] Build log validation tooling

### Phase 4: React Replay Viewer (Weeks 6-8)

- [ ] Build `GameLogProvider` context
- [ ] Implement board renderer (extend existing)
- [ ] Add playback controls
- [ ] Implement Fog of War toggle
- [ ] Add LLM reasoning panel
- [ ] Build turn timeline

### Phase 5: Testing & Benchmarking (Weeks 9-10)

- [ ] Run initial LLM tournaments
- [ ] Collect performance metrics
- [ ] Tune prompts based on results
- [ ] Document findings

---

## Appendix A: File Structure

```
catan-arena/
├── catan_arena/
│   ├── __init__.py
│   ├── cli.py                    # CLI entrypoint
│   ├── config.py                 # Configuration management
│   │
│   ├── orchestration/
│   │   ├── __init__.py
│   │   ├── graph.py              # LangGraph workflow
│   │   ├── nodes.py              # Node implementations
│   │   └── state.py              # State definitions
│   │
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── base.py               # Base PettingZoo wrapper
│   │   └── catan_pettingzoo.py   # Catan AEC environment
│   │
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py               # Abstract StateAdapter
│   │   ├── catan_state_adapter.py
│   │   └── catan_action_parser.py
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── providers.py          # LLM provider setup
│   │   └── prompts.py            # Prompt templates
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── game_log.py           # Log writing/reading
│   │   └── schemas/
│   │       └── game_log_v1.json  # JSON Schema
│   │
│   └── games/                    # Future game integrations
│       └── chess/
│           ├── __init__.py
│           ├── env.py
│           ├── adapter.py
│           └── parser.py
│
├── ui/
│   └── src/
│       ├── components/
│       │   └── replay/
│       │       ├── GameLogProvider.tsx
│       │       ├── BoardRenderer.tsx
│       │       ├── PlaybackControls.tsx
│       │       ├── ViewModeController.tsx
│       │       ├── LLMReasoningPanel.tsx
│       │       └── TurnTimeline.tsx
│       └── pages/
│           └── ReplayPage.tsx
│
├── tests/
│   ├── test_env.py
│   ├── test_adapter.py
│   ├── test_parser.py
│   └── test_orchestration.py
│
├── examples/
│   ├── run_tournament.py
│   └── analyze_logs.py
│
├── docs/
│   └── CATAN_ARENA_TDD.md        # This document
│
└── pyproject.toml
```

---

## Appendix B: Configuration Schema

```yaml
# config.yaml

arena:
  game_type: catan
  max_turns: 1000
  checkpoint_interval: 10  # Save state every N turns

  catan:
    map_type: BASE
    vps_to_win: 10
    num_players: 4

players:
  - id: claude_opus
    type: llm
    model: claude-3-opus-20240229
    config:
      temperature: 0.7
      max_tokens: 2048
    color: RED

  - id: gpt4_turbo
    type: llm
    model: gpt-4-turbo
    config:
      temperature: 0.7
      max_tokens: 2048
    color: BLUE

  - id: gemini_pro
    type: llm
    model: gemini-pro
    color: ORANGE

  - id: baseline_random
    type: random
    color: WHITE

storage:
  log_dir: ./game_logs
  checkpoint_dir: ./checkpoints
  compress_logs: true

logging:
  level: INFO
  format: json
```

---

**Document End**

*This TDD provides the architectural foundation for Catan-Arena MVP. Implementation details may evolve based on learnings during development.*
