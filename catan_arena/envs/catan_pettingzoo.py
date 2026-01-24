"""
PettingZoo AEC (Actor-Environment-Cycle) wrapper for Catanatron.

This module wraps the single-threaded Catanatron game engine into a
multi-agent environment where each agent (LLM) takes turns making decisions.
"""

from typing import Dict, List, Optional, Any, Tuple
import functools
import numpy as np

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from gymnasium import spaces

# Import from Catanatron
from catanatron.game import Game, TURNS_LIMIT
from catanatron.models.player import Player, Color
from catanatron.models.map import build_map, NUM_NODES
from catanatron.models.enums import (
    Action,
    ActionType,
    RESOURCES,
    DEVELOPMENT_CARDS,
)
from catanatron.state_functions import (
    player_key,
    get_player_freqdeck,
    get_actual_victory_points,
    get_visible_victory_points,
    player_num_resource_cards,
    player_num_dev_cards,
    get_played_dev_cards,
    get_longest_road_length,
    get_longest_road_color,
    get_largest_army,
    get_player_buildings,
)

# Reuse action space from existing Gymnasium environment
from catanatron.gym.envs.catanatron_env import (
    ACTIONS_ARRAY,
    ACTION_SPACE_SIZE,
    to_action_space,
    from_action_space,
    normalize_action,
)


class _ArenaPlayer(Player):
    """
    Placeholder player for Arena mode.

    All decisions come from external agents via the PettingZoo API,
    so this player's decide() method should never be called.
    """

    def __init__(self, color: Color):
        super().__init__(color, is_bot=False)

    def decide(self, game, playable_actions):
        raise RuntimeError(
            "Arena players should not auto-decide. "
            "Use CatanAECEnv.step() to provide actions."
        )


class CatanAECEnv(AECEnv):
    """
    PettingZoo AEC wrapper for Catanatron.

    Converts the Catanatron game engine into a multi-agent environment
    following the Actor-Environment-Cycle (AEC) API where agents take
    turns making decisions sequentially.

    Features:
    - 2-4 player support
    - Per-agent observation spaces (public + private views)
    - Action masking for valid moves only
    - Full game state serialization for replay

    Example:
        env = CatanAECEnv(num_players=4)
        env.reset()

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                action = None
            else:
                action = policy(observation, info["action_mask"])

            env.step(action)
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
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the Catan AEC environment.

        Args:
            num_players: Number of players (2-4)
            map_type: Map template ("BASE" or "MINI")
            vps_to_win: Victory points needed to win
            max_turns: Maximum turns before truncation
            render_mode: Rendering mode ("human", "json", or None)
        """
        super().__init__()

        assert 2 <= num_players <= 4, "Catan supports 2-4 players"

        self.num_players = num_players
        self.map_type = map_type
        self.vps_to_win = vps_to_win
        self.max_turns = max_turns
        self.render_mode = render_mode

        # Define agents with color mapping
        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self._color_list = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]
        self.agent_colors = {
            f"player_{i}": self._color_list[i]
            for i in range(num_players)
        }
        self.color_to_agent = {
            color: agent for agent, color in self.agent_colors.items()
        }

        # Action space: 290 discrete actions from Catanatron
        self._action_spaces = {
            agent: spaces.Discrete(ACTION_SPACE_SIZE)
            for agent in self.possible_agents
        }

        # Observation space: dict with public, private, and action_mask
        self._observation_spaces = {
            agent: spaces.Dict({
                "public": spaces.Dict({
                    "turn_number": spaces.Discrete(max_turns + 1),
                    "current_player": spaces.Discrete(num_players),
                    "robber_coordinate": spaces.Box(-3, 3, (3,), dtype=np.int32),
                    "bank_resources": spaces.Box(0, 19, (5,), dtype=np.int32),
                    "bank_dev_cards": spaces.Discrete(26),
                }),
                "private": spaces.Dict({
                    "hand_resources": spaces.Box(0, 19, (5,), dtype=np.int32),
                    "hand_dev_cards": spaces.Box(0, 5, (5,), dtype=np.int32),
                    "actual_victory_points": spaces.Discrete(20),
                }),
                "action_mask": spaces.MultiBinary(ACTION_SPACE_SIZE),
            })
            for agent in self.possible_agents
        }

        # Game state (initialized in reset)
        self.game: Optional[Game] = None
        self._agent_selector: Optional[agent_selector] = None
        self.turn_count = 0

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        """Get observation space for an agent."""
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        """Get action space for an agent."""
        return self._action_spaces[agent]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> None:
        """
        Reset the environment for a new game.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
        """
        # Create placeholder players
        colors = list(self.agent_colors.values())
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
        self.agents = list(self.possible_agents)
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        # Sync with game's current player
        self._sync_agent_selection()

        # Initialize per-agent state
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}

        self.turn_count = 0

    def _sync_agent_selection(self) -> None:
        """Sync PettingZoo agent selection with Catanatron's current player."""
        if self.game is None or self.game.winning_color() is not None:
            return

        current_color = self.game.state.current_color()
        current_agent = self.color_to_agent.get(current_color)

        if current_agent is None:
            return

        # Advance selector until it matches game's current player
        max_iterations = self.num_players + 1
        for _ in range(max_iterations):
            if self.agent_selection == current_agent:
                break
            self.agent_selection = self._agent_selector.next()

    def step(self, action: Optional[int]) -> None:
        """
        Execute an action for the current agent.

        Args:
            action: Integer action index, or None if agent is done
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        color = self.agent_colors[agent]

        # Validate it's this agent's turn
        if self.game.state.current_color() != color:
            # Not this agent's turn - skip with zero reward
            self.rewards[agent] = 0
            self._advance_agent()
            return

        # Convert action integer to Catanatron Action
        if action is not None:
            catan_action = self._action_int_to_catan(action)

            if catan_action is None:
                # Invalid action - apply small penalty
                self.rewards[agent] = -0.01
                self.infos[agent]["invalid_action"] = True
                self.infos[agent]["action_attempted"] = action
            else:
                # Execute valid action
                try:
                    action_record = self.game.execute(catan_action)
                    self.rewards[agent] = 0
                    self.infos[agent]["action_executed"] = {
                        "action_type": catan_action.action_type.value,
                        "value": catan_action.value,
                    }
                    self.infos[agent]["action_record"] = action_record
                except Exception as e:
                    self.rewards[agent] = -0.01
                    self.infos[agent]["execution_error"] = str(e)

        # Check for game end
        winner = self.game.winning_color()
        if winner is not None:
            self._handle_game_end(winner)
        elif self.turn_count >= self.max_turns:
            self._handle_truncation()
        else:
            # Track turn count (increment when current player changes)
            if self.game.state.current_color() != color:
                self.turn_count += 1

        # Advance to next agent
        self._advance_agent()

    def _advance_agent(self) -> None:
        """Advance to the next agent."""
        self.agent_selection = self._agent_selector.next()
        self._sync_agent_selection()

    def _action_int_to_catan(self, action_int: int) -> Optional[Action]:
        """Convert integer action to Catanatron Action."""
        try:
            return from_action_space(action_int, self.game.playable_actions)
        except (ValueError, AssertionError):
            return None

    def _handle_game_end(self, winner: Color) -> None:
        """Set termination flags when game ends with a winner."""
        winner_agent = self.color_to_agent.get(winner)

        for agent in self.agents:
            self.terminations[agent] = True
            if agent == winner_agent:
                self.rewards[agent] = 1.0
            else:
                self.rewards[agent] = -1.0

    def _handle_truncation(self) -> None:
        """Handle max turns reached (truncation)."""
        # Find leader by VP
        vp_scores = {}
        for agent in self.agents:
            color = self.agent_colors[agent]
            vp_scores[agent] = get_actual_victory_points(self.game.state, color)

        max_vp = max(vp_scores.values())

        for agent in self.agents:
            self.truncations[agent] = True
            # Partial reward based on VP ranking
            if vp_scores[agent] == max_vp:
                self.rewards[agent] = 0.5  # Leader bonus
            else:
                self.rewards[agent] = -0.5

    def observe(self, agent: str) -> Dict[str, Any]:
        """
        Get observation for a specific agent.

        Returns both public state (visible to all) and private state
        (visible only to this agent).
        """
        color = self.agent_colors[agent]

        # Build observation dict
        observation = {
            "public": self._get_public_observation(),
            "private": self._get_private_observation(color),
            "action_mask": self._get_action_mask(color),
        }

        return observation

    def _get_public_observation(self) -> Dict[str, Any]:
        """Extract publicly visible game state."""
        state = self.game.state
        board = state.board

        # Get longest road and largest army holders
        longest_road_color = get_longest_road_color(state)
        largest_army_color, army_size = get_largest_army(state)

        return {
            "turn_number": self.turn_count,
            "current_player": state.colors.index(state.current_color()),
            "current_prompt": state.current_prompt.value if state.current_prompt else None,
            "is_initial_build_phase": state.is_initial_build_phase,
            "robber_coordinate": list(board.robber_coordinate),
            "bank_resources": list(state.resource_freqdeck),
            "bank_dev_cards": len(state.development_listdeck),
            "longest_road_owner": (
                self.color_to_agent.get(longest_road_color)
                if longest_road_color else None
            ),
            "largest_army_owner": (
                self.color_to_agent.get(largest_army_color)
                if largest_army_color else None
            ),
            "largest_army_size": army_size or 0,
            "player_summaries": {
                self.color_to_agent[color]: self._get_player_summary(color)
                for color in state.colors
            },
            "board": self._serialize_board(),
        }

    def _get_player_summary(self, color: Color) -> Dict[str, Any]:
        """Get publicly visible information about a player."""
        state = self.game.state
        key = player_key(state, color)

        return {
            "color": color.value,
            "victory_points_visible": get_visible_victory_points(state, color),
            "resource_count": player_num_resource_cards(state, color),
            "dev_card_count": player_num_dev_cards(state, color),
            "knights_played": get_played_dev_cards(state, color, "KNIGHT"),
            "longest_road_length": get_longest_road_length(state, color),
            "has_longest_road": state.player_state.get(f"{key}_HAS_ROAD", False),
            "has_largest_army": state.player_state.get(f"{key}_HAS_ARMY", False),
            "settlements_left": state.player_state.get(f"{key}_SETTLEMENTS_AVAILABLE", 0),
            "cities_left": state.player_state.get(f"{key}_CITIES_AVAILABLE", 0),
            "roads_left": state.player_state.get(f"{key}_ROADS_AVAILABLE", 0),
        }

    def _get_private_observation(self, color: Color) -> Dict[str, Any]:
        """Extract private state visible only to one player."""
        state = self.game.state
        key = player_key(state, color)

        return {
            "hand_resources": {
                resource: state.player_state.get(f"{key}_{resource}_IN_HAND", 0)
                for resource in RESOURCES
            },
            "hand_dev_cards": {
                card: state.player_state.get(f"{key}_{card}_IN_HAND", 0)
                for card in DEVELOPMENT_CARDS
            },
            "actual_victory_points": get_actual_victory_points(state, color),
            "can_play_dev_card": not state.player_state.get(
                f"{key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN", False
            ),
            "has_rolled": state.player_state.get(f"{key}_HAS_ROLLED", False),
        }

    def _get_action_mask(self, color: Color) -> np.ndarray:
        """Generate action mask for valid actions."""
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)

        # Only mask if it's this player's turn
        if self.game.state.current_color() != color:
            return mask

        for action in self.game.playable_actions:
            try:
                action_int = to_action_space(action)
                mask[action_int] = 1
            except (ValueError, IndexError):
                # Action not in standard action space
                pass

        return mask

    def _serialize_board(self) -> Dict[str, Any]:
        """Serialize board state to dict format."""
        board = self.game.state.board

        # Tiles
        tiles = []
        for coord, tile in board.map.tiles.items():
            tile_data = {
                "coordinate": list(coord),
                "type": type(tile).__name__,
            }
            if hasattr(tile, "resource"):
                tile_data["resource"] = tile.resource
            if hasattr(tile, "number"):
                tile_data["number"] = tile.number
            tiles.append(tile_data)

        # Buildings (settlements and cities)
        buildings = []
        for node_id, building in board.buildings.items():
            if building:
                buildings.append({
                    "node_id": node_id,
                    "color": building[0].value,
                    "type": building[1],
                })

        # Roads
        roads = []
        for edge, color in board.roads.items():
            roads.append({
                "edge": list(edge),
                "color": color.value,
            })

        return {
            "tiles": tiles,
            "buildings": buildings,
            "roads": roads,
            "robber_coordinate": list(board.robber_coordinate),
        }

    def get_valid_actions(self, agent: str) -> List[int]:
        """Get list of valid action indices for an agent."""
        color = self.agent_colors[agent]
        mask = self._get_action_mask(color)
        return [i for i, valid in enumerate(mask) if valid]

    def get_playable_actions(self) -> List[Action]:
        """Get the current list of playable Catanatron actions."""
        return list(self.game.playable_actions) if self.game else []

    def render(self) -> Optional[Dict]:
        """Render the environment."""
        if self.render_mode == "json":
            return self._get_public_observation()
        elif self.render_mode == "human":
            # Print basic game state
            state = self.game.state
            print(f"\nTurn {self.turn_count}")
            print(f"Current player: {state.current_color().value}")
            print(f"Prompt: {state.current_prompt}")
            for color in state.colors:
                vp = get_actual_victory_points(state, color)
                print(f"  {color.value}: {vp} VP")
        return None

    def close(self) -> None:
        """Clean up resources."""
        pass

    def state(self) -> Dict[str, Any]:
        """Get the full game state (for serialization)."""
        return {
            "game_state": self._get_public_observation(),
            "private_states": {
                agent: self._get_private_observation(self.agent_colors[agent])
                for agent in self.agents
            },
            "turn_count": self.turn_count,
            "terminations": dict(self.terminations),
            "truncations": dict(self.truncations),
            "rewards": dict(self.rewards),
        }


def create_catan_env(
    num_players: int = 4,
    map_type: str = "BASE",
    vps_to_win: int = 10,
    max_turns: int = 1000,
    render_mode: Optional[str] = None,
) -> CatanAECEnv:
    """
    Factory function to create a Catan AEC environment.

    Args:
        num_players: Number of players (2-4)
        map_type: Map template ("BASE" or "MINI")
        vps_to_win: Victory points needed to win
        max_turns: Maximum turns before truncation
        render_mode: Rendering mode ("human", "json", or None)

    Returns:
        CatanAECEnv instance
    """
    return CatanAECEnv(
        num_players=num_players,
        map_type=map_type,
        vps_to_win=vps_to_win,
        max_turns=max_turns,
        render_mode=render_mode,
    )
