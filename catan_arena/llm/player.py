"""
LLM-based player for Catan-Arena.

This player uses a Large Language Model to make strategic decisions
in Settlers of Catan, integrating with the Catanatron Player interface.
"""

import time
from typing import List, Dict, Any, Optional, Iterable
from dataclasses import dataclass, field

from catanatron.models.player import Player, Color
from catanatron.models.enums import Action
from catanatron.game import Game

from catan_arena.adapters.catan_state_adapter import CatanStateAdapter, CatanPromptConfig
from catan_arena.adapters.catan_action_parser import CatanActionParser
from catan_arena.llm.providers import get_llm_for_model, resolve_model_alias


@dataclass
class LLMDecisionRecord:
    """Record of an LLM decision for logging and analysis."""

    model_name: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    reasoning: str = ""
    raw_output: str = ""
    parsed_action: Optional[Dict[str, Any]] = None
    parsed_successfully: bool = False
    parse_error: Optional[str] = None
    action_executed: Optional[Action] = None


class LLMPlayer(Player):
    """
    LLM-based Catan player.

    Uses a Large Language Model to analyze game state and make strategic
    decisions. Integrates with the Catanatron Player interface so it can
    be used directly with the game engine.

    Example:
        from catanatron.models.player import Color
        from catan_arena.llm.player import LLMPlayer

        player = LLMPlayer(
            color=Color.RED,
            model_name="claude-3-opus-20240229",
        )

        # Use in a game
        from catanatron.game import Game
        game = Game([player, RandomPlayer(Color.BLUE)])
        game.play()
    """

    def __init__(
        self,
        color: Color,
        model_name: str = "claude-3-opus-20240229",
        llm_config: Optional[Dict[str, Any]] = None,
        adapter_config: Optional[CatanPromptConfig] = None,
        fallback_to_random: bool = True,
        record_decisions: bool = True,
    ):
        """
        Initialize the LLM player.

        Args:
            color: Player color
            model_name: LLM model identifier (or alias)
            llm_config: LLM configuration (temperature, max_tokens, etc.)
            adapter_config: State adapter configuration
            fallback_to_random: If True, use random action on LLM/parse failure
            record_decisions: If True, record decision details for logging
        """
        super().__init__(color, is_bot=True)

        # Resolve model alias
        self.model_name = resolve_model_alias(model_name)
        self.llm_config = llm_config or {}

        # Initialize components
        self.adapter = CatanStateAdapter(adapter_config or CatanPromptConfig())
        self.parser = CatanActionParser(fallback_to_random=fallback_to_random)
        self.llm = get_llm_for_model(self.model_name, self.llm_config)

        # State
        self.fallback_to_random = fallback_to_random
        self.record_decisions = record_decisions
        self.decision_history: List[LLMDecisionRecord] = []
        self.turn_history: List[Dict[str, Any]] = []

    def decide(self, game: Game, playable_actions: Iterable[Action]) -> Action:
        """
        Make a decision using the LLM.

        Args:
            game: Current game state (read-only)
            playable_actions: List of valid actions

        Returns:
            Chosen Action
        """
        playable_actions = list(playable_actions)

        # Short-circuit if only one action
        if len(playable_actions) == 1:
            return playable_actions[0]

        # Build observation from game state
        public_state = self._extract_public_state(game)
        private_state = self._extract_private_state(game)
        valid_actions = self.adapter.format_valid_actions_for_prompt(playable_actions)

        # Generate prompt
        prompt = self.adapter.state_to_prompt(
            public_state=public_state,
            private_state=private_state,
            valid_actions=valid_actions,
            turn_history=self.turn_history[-10:],  # Last 10 turns
        )

        # Call LLM
        start_time = time.time()
        try:
            response = self.llm.invoke([
                {"role": "system", "content": self.adapter.format_system_prompt()},
                {"role": "user", "content": prompt},
            ])
            raw_output = response.content
            latency_ms = int((time.time() - start_time) * 1000)

            # Extract token usage if available
            token_usage = getattr(response, "usage_metadata", {}) or {}
            prompt_tokens = token_usage.get("input_tokens", 0)
            completion_tokens = token_usage.get("output_tokens", 0)
            total_tokens = token_usage.get("total_tokens", prompt_tokens + completion_tokens)

        except Exception as e:
            # LLM call failed - use fallback
            raw_output = f"LLM Error: {str(e)}"
            latency_ms = int((time.time() - start_time) * 1000)
            prompt_tokens = completion_tokens = total_tokens = 0

            if self.fallback_to_random:
                import random
                action = random.choice(playable_actions)
                self._record_decision(
                    raw_output=raw_output,
                    latency_ms=latency_ms,
                    parsed_successfully=False,
                    parse_error=str(e),
                    action=action,
                )
                return action
            raise

        # Parse LLM output
        reasoning = self.parser.extract_reasoning(raw_output)

        try:
            parsed = self.parser.parse(raw_output, valid_actions)
            parsed_successfully = True
            parse_error = None

            # Find matching action
            action = self._match_action(parsed, playable_actions)

            if action is None:
                # Parsed but couldn't match - use first playable
                parsed_successfully = False
                parse_error = "Could not match parsed action to playable actions"
                action = playable_actions[0]

        except Exception as e:
            parsed = None
            parsed_successfully = False
            parse_error = str(e)

            if self.fallback_to_random:
                import random
                action = random.choice(playable_actions)
            else:
                raise

        # Record decision
        self._record_decision(
            raw_output=raw_output,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            reasoning=reasoning,
            parsed=parsed,
            parsed_successfully=parsed_successfully,
            parse_error=parse_error,
            action=action,
        )

        return action

    def _match_action(
        self,
        parsed: Dict[str, Any],
        playable_actions: List[Action],
    ) -> Optional[Action]:
        """Match parsed action dict to a Catanatron Action."""
        from catanatron.models.enums import ActionType

        action_type_str = parsed.get("action_type")
        value = parsed.get("value")

        try:
            action_type = ActionType[action_type_str]
        except KeyError:
            return None

        # Find matching playable action
        for action in playable_actions:
            if action.action_type != action_type:
                continue

            # Match value
            if self._values_match(value, action.value, action_type):
                return action

        return None

    def _values_match(
        self,
        parsed_value: Any,
        action_value: Any,
        action_type: Any,
    ) -> bool:
        """Check if parsed value matches action value."""
        from catanatron.models.enums import ActionType

        # Null values
        if parsed_value is None and action_value is None:
            return True

        if parsed_value is None or action_value is None:
            return False

        # Direct equality
        if parsed_value == action_value:
            return True

        # List/tuple equivalence
        if isinstance(parsed_value, (list, tuple)) and isinstance(action_value, (list, tuple)):
            if list(parsed_value) == list(action_value):
                return True

            # BUILD_ROAD: edge order doesn't matter
            if action_type == ActionType.BUILD_ROAD:
                if len(parsed_value) == 2 and len(action_value) == 2:
                    if tuple(sorted(parsed_value)) == tuple(sorted(action_value)):
                        return True

        return False

    def _extract_public_state(self, game: Game) -> Dict[str, Any]:
        """Extract public state from game."""
        from catanatron.state_functions import (
            get_visible_victory_points,
            player_num_resource_cards,
            player_num_dev_cards,
            get_played_dev_cards,
            get_longest_road_length,
            get_longest_road_color,
            get_largest_army,
            player_key,
        )

        state = game.state
        board = state.board

        # Get longest road and largest army
        longest_road_color = get_longest_road_color(state)
        largest_army_color, army_size = get_largest_army(state)

        # Build player summaries
        player_summaries = {}
        for i, color in enumerate(state.colors):
            key = player_key(state, color)
            player_summaries[f"player_{i}"] = {
                "color": color.value,
                "victory_points_visible": get_visible_victory_points(state, color),
                "resource_count": player_num_resource_cards(state, color),
                "dev_card_count": player_num_dev_cards(state, color),
                "knights_played": get_played_dev_cards(state, color, "KNIGHT"),
                "longest_road_length": get_longest_road_length(state, color),
                "has_longest_road": state.player_state.get(f"{key}_HAS_ROAD", False),
                "has_largest_army": state.player_state.get(f"{key}_HAS_ARMY", False),
            }

        # Serialize board
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

        buildings = []
        for node_id, building in board.buildings.items():
            if building:
                buildings.append({
                    "node_id": node_id,
                    "color": building[0].value,
                    "type": building[1],
                })

        roads = []
        for edge, color in board.roads.items():
            roads.append({
                "edge": list(edge),
                "color": color.value,
            })

        return {
            "turn_number": state.num_turns,
            "current_prompt": state.current_prompt.value if state.current_prompt else None,
            "is_initial_build_phase": state.is_initial_build_phase,
            "robber_coordinate": list(board.robber_coordinate),
            "bank_resources": list(state.resource_freqdeck),
            "bank_dev_cards": len(state.development_listdeck),
            "longest_road_owner": longest_road_color.value if longest_road_color else None,
            "largest_army_owner": largest_army_color.value if largest_army_color else None,
            "player_summaries": player_summaries,
            "board": {
                "tiles": tiles,
                "buildings": buildings,
                "roads": roads,
                "robber_coordinate": list(board.robber_coordinate),
            },
        }

    def _extract_private_state(self, game: Game) -> Dict[str, Any]:
        """Extract private state for this player."""
        from catanatron.models.enums import RESOURCES, DEVELOPMENT_CARDS
        from catanatron.state_functions import (
            player_key,
            get_actual_victory_points,
        )

        state = game.state
        key = player_key(state, self.color)

        return {
            "hand_resources": {
                resource: state.player_state.get(f"{key}_{resource}_IN_HAND", 0)
                for resource in RESOURCES
            },
            "hand_dev_cards": {
                card: state.player_state.get(f"{key}_{card}_IN_HAND", 0)
                for card in DEVELOPMENT_CARDS
            },
            "actual_victory_points": get_actual_victory_points(state, self.color),
            "can_play_dev_card": not state.player_state.get(
                f"{key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN", False
            ),
            "has_rolled": state.player_state.get(f"{key}_HAS_ROLLED", False),
        }

    def _record_decision(
        self,
        raw_output: str,
        latency_ms: int,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        reasoning: str = "",
        parsed: Optional[Dict[str, Any]] = None,
        parsed_successfully: bool = False,
        parse_error: Optional[str] = None,
        action: Optional[Action] = None,
    ):
        """Record decision for logging."""
        if not self.record_decisions:
            return

        record = LLMDecisionRecord(
            model_name=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            reasoning=reasoning,
            raw_output=raw_output,
            parsed_action=parsed,
            parsed_successfully=parsed_successfully,
            parse_error=parse_error,
            action_executed=action,
        )
        self.decision_history.append(record)

    def add_turn_to_history(self, turn_record: Dict[str, Any]):
        """Add a turn record to history for context."""
        self.turn_history.append(turn_record)

    def get_last_decision(self) -> Optional[LLMDecisionRecord]:
        """Get the most recent decision record."""
        return self.decision_history[-1] if self.decision_history else None

    def reset_state(self):
        """Reset player state between games."""
        self.decision_history = []
        self.turn_history = []

    def __repr__(self):
        return f"LLMPlayer({self.model_name}):{self.color.value}"
