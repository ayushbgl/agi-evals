"""
Catan-specific state adapter for converting game state to LLM prompts.

This adapter converts the complex Catan game state (19-tile hex board,
player hands, buildings, etc.) into a text prompt that LLMs can understand
and reason about strategically.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from catan_arena.adapters.base import StateAdapter


# Dice number probabilities
DICE_PROBABILITIES = {
    2: 1/36, 3: 2/36, 4: 3/36, 5: 4/36, 6: 5/36,
    7: 6/36,
    8: 5/36, 9: 4/36, 10: 3/36, 11: 2/36, 12: 1/36,
}


@dataclass
class CatanPromptConfig:
    """Configuration for Catan prompt generation."""

    board_representation: str = "ascii"  # "ascii", "list", "both"
    include_probabilities: bool = True
    max_history_turns: int = 10
    include_valid_actions: bool = True
    compact_mode: bool = False  # Shorter prompts for smaller context models


class CatanStateAdapter(StateAdapter):
    """
    Converts Catan game state to natural language prompts.

    Supports multiple board representation strategies:
    1. ASCII: Visual hex grid (better for spatial reasoning)
    2. List: Structured data (more precise, longer)
    3. Both: Combination for maximum clarity
    """

    RESOURCE_SYMBOLS = {
        "WOOD": "W",
        "BRICK": "B",
        "SHEEP": "S",
        "WHEAT": "H",  # H for Harvest
        "ORE": "O",
    }

    RESOURCE_NAMES = {
        "WOOD": "Wood",
        "BRICK": "Brick",
        "SHEEP": "Sheep",
        "WHEAT": "Wheat",
        "ORE": "Ore",
    }

    def __init__(self, config: Optional[CatanPromptConfig] = None):
        """
        Initialize the Catan state adapter.

        Args:
            config: Prompt generation configuration
        """
        self.config = config or CatanPromptConfig()

    def state_to_prompt(
        self,
        public_state: Dict[str, Any],
        private_state: Dict[str, Any],
        valid_actions: List[Dict[str, Any]],
        turn_history: Optional[List[Dict[str, Any]]] = None,
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
            self._turn_context_section(public_state),
            self._board_section(public_state),
            self._your_state_section(private_state, public_state),
            self._opponents_section(public_state),
        ]

        # Add pending trade section if there's a trade offer
        if public_state.get("pending_trade"):
            sections.append(self._trade_section(public_state["pending_trade"]))

        if turn_history:
            sections.append(self._history_section(turn_history))

        sections.append(self._actions_section(valid_actions))
        sections.append(self._output_instructions())

        return "\n\n".join(sections)

    def format_system_prompt(self) -> str:
        """Return the system prompt for the LLM."""
        return """You are an expert Settlers of Catan player. You analyze game states strategically and make optimal decisions to win the game. You always respond with valid JSON containing your chosen action and reasoning."""

    def _system_context(self) -> str:
        """Game rules and context."""
        if self.config.compact_mode:
            return """# SETTLERS OF CATAN

Goal: First to 10 Victory Points wins.
VP Sources: Settlement=1, City=2, Longest Road(5+)=2, Largest Army(3+)=2, VP cards=1
Costs: Road=W+B, Settlement=W+B+S+H, City=2H+3O, Dev Card=S+H+O"""

        return """# SETTLERS OF CATAN - AI PLAYER

You are playing Settlers of Catan. Your goal is to reach 10 Victory Points first.

## Victory Point Sources:
- Settlement: 1 VP
- City: 2 VP (upgrade from settlement)
- Longest Road (5+ roads in a row): 2 VP
- Largest Army (3+ knights played): 2 VP
- Victory Point cards: 1 VP each (hidden until winning)

## Building Costs:
- Road: 1 Wood + 1 Brick
- Settlement: 1 Wood + 1 Brick + 1 Sheep + 1 Wheat
- City (upgrade): 2 Wheat + 3 Ore
- Development Card: 1 Sheep + 1 Wheat + 1 Ore

## Key Rules:
- You must roll dice at the start of your turn (unless playing a Knight first)
- You can only build on your turn after rolling
- Settlements must be at least 2 roads apart
- Rolling 7: Move robber, players with 8+ cards discard half
- You can trade with the bank (4:1) or use ports (3:1 or 2:1)
- Development cards bought this turn cannot be played until next turn"""

    def _turn_context_section(self, public_state: Dict[str, Any]) -> str:
        """Generate turn context with phase and dice info."""
        lines = ["## CURRENT TURN"]

        phase = public_state.get("phase", "MAIN")
        current_player = public_state.get("current_player", "?")
        turn_number = public_state.get("turn_number", 0)

        lines.append(f"Turn: {turn_number}")
        lines.append(f"Current Player: {current_player}")
        lines.append(f"Phase: {phase}")

        if phase == "INITIAL_BUILD":
            lines.append("")
            lines.append("📋 INITIAL PLACEMENT PHASE - No dice rolling.")
            lines.append("   Place your settlement and road to start the game.")
        else:
            # Show dice roll if available
            dice_info = public_state.get("last_dice_roll")
            if dice_info:
                dice = dice_info.get("dice", [])
                total = dice_info.get("total", sum(dice) if dice else 0)
                lines.append("")
                lines.append(f"🎲 DICE ROLLED: {dice[0]} + {dice[1]} = {total}")
                if total == 7:
                    lines.append("   ⚠️ SEVEN! Robber must be moved. Players with 8+ cards discard half.")

        return "\n".join(lines)

    def _board_section(self, public_state: Dict[str, Any]) -> str:
        """Generate the board representation."""
        board = public_state.get("board", {})

        if self.config.board_representation == "ascii":
            return self._ascii_board(board, public_state)
        elif self.config.board_representation == "list":
            return self._list_board(board, public_state)
        else:  # "both"
            return self._ascii_board(board, public_state) + "\n\n" + self._list_board(board, public_state)

    def _ascii_board(self, board: Dict, public_state: Dict) -> str:
        """Generate ASCII representation of the hex board."""
        lines = ["## BOARD STATE"]

        if not self.config.compact_mode:
            lines.append("```")
            lines.append("Legend: W=Wood B=Brick S=Sheep H=Wheat O=Ore D=Desert")
            lines.append("        Numbers show dice roll needed. *=Robber location")
            lines.append("```")

        # Build tile information
        tiles = board.get("tiles", [])
        robber_coord = tuple(public_state.get("robber_coordinate", [0, 0, 0]))

        lines.append("\n### Resource Tiles:")
        for tile in sorted(tiles, key=lambda t: t.get("coordinate", [0, 0, 0])):
            coord = tuple(tile.get("coordinate", [0, 0, 0]))
            tile_type = tile.get("type", "")

            if "Land" in tile_type and tile.get("resource"):
                resource = tile.get("resource", "?")
                number = tile.get("number", 0)
                symbol = self.RESOURCE_SYMBOLS.get(resource, "?")
                robber_mark = " *ROBBER*" if coord == robber_coord else ""
                prob = DICE_PROBABILITIES.get(number, 0) * 100

                if self.config.include_probabilities:
                    lines.append(f"  {coord}: {symbol}({resource}) #{number} ({prob:.1f}%){robber_mark}")
                else:
                    lines.append(f"  {coord}: {symbol}({resource}) #{number}{robber_mark}")
            elif "Desert" in str(tile_type) or tile.get("resource") is None:
                robber_mark = " *ROBBER*" if coord == robber_coord else ""
                lines.append(f"  {coord}: DESERT{robber_mark}")

        # Buildings
        buildings = board.get("buildings", [])
        if buildings:
            lines.append("\n### Buildings:")
            by_color = {}
            for b in buildings:
                color = b.get("color", "?")
                if color not in by_color:
                    by_color[color] = []
                by_color[color].append(f"{b.get('type', '?')} at node {b.get('node_id', '?')}")

            for color, items in by_color.items():
                lines.append(f"  {color}: {', '.join(items)}")

        # Roads
        roads = board.get("roads", [])
        if roads:
            lines.append("\n### Roads:")
            by_color = {}
            for r in roads:
                color = r.get("color", "?")
                if color not in by_color:
                    by_color[color] = 0
                by_color[color] += 1

            for color, count in by_color.items():
                lines.append(f"  {color}: {count} roads")

        return "\n".join(lines)

    def _list_board(self, board: Dict, public_state: Dict) -> str:
        """Generate structured list representation."""
        lines = ["## BOARD DETAILS"]

        tiles = board.get("tiles", [])
        robber_coord = tuple(public_state.get("robber_coordinate", [0, 0, 0]))

        # Resource production summary
        lines.append("\n### Production by Number:")
        by_number = {}
        for tile in tiles:
            if tile.get("resource") and tile.get("number"):
                num = tile["number"]
                if num not in by_number:
                    by_number[num] = []
                by_number[num].append(tile["resource"])

        for num in sorted(by_number.keys()):
            prob = DICE_PROBABILITIES.get(num, 0) * 100
            resources = ", ".join(by_number[num])
            lines.append(f"  {num} ({prob:.1f}%): {resources}")

        return "\n".join(lines)

    def _your_state_section(self, private_state: Dict, public_state: Dict) -> str:
        """Format the player's private state (their hand)."""
        lines = ["## YOUR STATE (Private - only you can see this)"]

        # Resources
        hand = private_state.get("hand_resources", {})
        total_resources = sum(hand.values())
        lines.append(f"\n### Your Hand ({total_resources} resource cards):")

        for resource, count in hand.items():
            if count > 0:
                name = self.RESOURCE_NAMES.get(resource, resource)
                lines.append(f"  - {name}: {count}")

        if total_resources == 0:
            lines.append("  (empty)")

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

        # Status flags
        if private_state.get("has_rolled"):
            lines.append("  - You have already rolled this turn")
        if not private_state.get("can_play_dev_card", True):
            lines.append("  - You have already played a dev card this turn")

        return "\n".join(lines)

    def _opponents_section(self, public_state: Dict) -> str:
        """Format publicly visible information about opponents."""
        lines = ["## OTHER PLAYERS (Public Information)"]

        summaries = public_state.get("player_summaries", {})
        for player_id, info in summaries.items():
            lines.append(f"\n### {player_id} ({info.get('color', '?')}):")
            lines.append(f"  - Visible VP: {info.get('victory_points_visible', 0)}")
            lines.append(f"  - Resource cards: {info.get('resource_count', 0)} (hidden)")
            lines.append(f"  - Dev cards: {info.get('dev_card_count', 0)} (hidden)")
            lines.append(f"  - Knights played: {info.get('knights_played', 0)}")
            lines.append(f"  - Longest road: {info.get('longest_road_length', 0)}")

            if info.get("has_longest_road"):
                lines.append("  - ** HAS LONGEST ROAD BONUS (+2 VP) **")
            if info.get("has_largest_army"):
                lines.append("  - ** HAS LARGEST ARMY BONUS (+2 VP) **")

        return "\n".join(lines)

    def _trade_section(self, trade_info: Dict) -> str:
        """Format pending trade offer information."""
        lines = ["## ⚠️ PENDING TRADE OFFER"]
        lines.append("")
        lines.append("Another player has offered a trade. You must ACCEPT or REJECT.")
        lines.append("")

        offering = trade_info.get("offering", {})
        requesting = trade_info.get("requesting", {})

        # What they're offering (you would receive)
        offer_items = [f"{count} {res}" for res, count in offering.items() if count > 0]
        if offer_items:
            lines.append(f"**They offer (you receive):** {', '.join(offer_items)}")
        else:
            lines.append("**They offer:** Nothing")

        # What they're requesting (you would give)
        request_items = [f"{count} {res}" for res, count in requesting.items() if count > 0]
        if request_items:
            lines.append(f"**They request (you give):** {', '.join(request_items)}")
        else:
            lines.append("**They request:** Nothing")

        lines.append("")
        lines.append("Consider: Do you have the resources? Is this trade beneficial to you?")

        return "\n".join(lines)

    def _history_section(self, turn_history: List[Dict]) -> str:
        """Format recent game history."""
        if not turn_history:
            return "## RECENT HISTORY\nNo previous turns."

        max_turns = self.config.max_history_turns
        recent = turn_history[-max_turns:] if len(turn_history) > max_turns else turn_history

        lines = [f"## RECENT HISTORY (Last {len(recent)} actions)"]

        for i, turn in enumerate(recent, 1):
            # Support both old format (player_id + action dict) and new format (player + action_type)
            player = turn.get("player") or turn.get("player_id", "?")

            # Handle new flat format
            action_type = turn.get("action_type")
            if action_type is None:
                # Old nested format
                action = turn.get("action", {})
                action_type = action.get("action_type", "?")
                value = action.get("value")
            else:
                value = turn.get("value")

            # Format based on action type
            if action_type == "ROLL":
                dice = turn.get("value") or []
                total = turn.get("total") or (sum(dice) if dice else "?")
                lines.append(f"  {i}. {player} rolled {total} {dice}")
            elif action_type in ["BUILD_SETTLEMENT", "BUILD_CITY", "BUILD_ROAD"]:
                lines.append(f"  {i}. {player} {action_type.replace('_', ' ').lower()} at {value}")
            elif action_type == "BUY_DEVELOPMENT_CARD":
                lines.append(f"  {i}. {player} bought a development card")
            elif action_type == "END_TURN":
                lines.append(f"  {i}. {player} ended turn")
            else:
                action_desc = action_type.replace('_', ' ').lower()
                if value:
                    lines.append(f"  {i}. {player} {action_desc} {value}")
                else:
                    lines.append(f"  {i}. {player} {action_desc}")

        return "\n".join(lines)

    def _actions_section(self, valid_actions: List[Dict]) -> str:
        """Format available actions."""
        lines = ["## AVAILABLE ACTIONS"]
        lines.append("You must choose ONE of these actions:")
        lines.append("")

        # Group by action type
        by_type = {}
        for action in valid_actions:
            atype = action.get("action_type", "UNKNOWN")
            if atype not in by_type:
                by_type[atype] = []
            by_type[atype].append(action)

        for atype, actions in by_type.items():
            if len(actions) == 1 and actions[0].get("value") is None:
                lines.append(f"- {atype}")
            else:
                lines.append(f"- {atype}:")
                # Show all options (needed for LLM to make informed decisions)
                for action in actions:
                    value = action.get("value")
                    if value is not None:
                        # Format maritime trade more clearly
                        if atype == "MARITIME_TRADE" and isinstance(value, (list, tuple)) and len(value) == 5:
                            give = [v for v in value[:4] if v is not None]
                            receive = value[4]
                            ratio = len(give)
                            give_str = give[0] if give else "?"
                            lines.append(f"    - Give {ratio}x {give_str} → Get 1x {receive}")
                            lines.append(f"      (value: {list(value)})")
                        elif atype == "OFFER_TRADE" and isinstance(value, (list, tuple)) and len(value) == 10:
                            # Format player trade
                            resources = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
                            offer = [f"{value[i]}x{resources[i]}" for i in range(5) if value[i] > 0]
                            request = [f"{value[i+5]}x{resources[i]}" for i in range(5) if value[i+5] > 0]
                            lines.append(f"    - Offer: {', '.join(offer) or 'nothing'} → Want: {', '.join(request) or 'nothing'}")
                            lines.append(f"      (value: {list(value)})")
                        else:
                            lines.append(f"    - value: {value}")

        return "\n".join(lines)

    def _output_instructions(self) -> str:
        """Instructions for LLM output format."""
        if self.config.compact_mode:
            return """## RESPONSE FORMAT

Think step by step, then output JSON:

```json
{"action_type": "BUILD_SETTLEMENT", "value": 23, "rationale": "Why this move"}
```

MARITIME_TRADE: [give,give,give,give,receive] - e.g. ["WOOD","WOOD","WOOD","WOOD","ORE"] = give 4 wood, get 1 ore
OFFER_TRADE: [offer_W,B,S,H,O, request_W,B,S,H,O] - e.g. [1,0,0,0,0, 0,0,0,1,0] = offer 1 wood, want 1 wheat"""

        return """## YOUR RESPONSE

Think through your decision step by step, considering:
1. Your current resources and what you can build
2. Your progress toward victory (settlements, cities, roads, dev cards)
3. What opponents are doing and their VP counts
4. The probability of getting resources you need
5. The dice roll result (if shown above)

Then output your chosen action as a JSON code block:

```json
{
  "action_type": "BUILD_SETTLEMENT",
  "value": 23,
  "rationale": "Brief explanation of why this action"
}
```

### Action Types and Values:

| Action Type | Value Format | Example |
|-------------|--------------|---------|
| END_TURN | null | {"action_type": "END_TURN", "value": null} |
| BUILD_ROAD | [node1, node2] | {"action_type": "BUILD_ROAD", "value": [12, 13]} |
| BUILD_SETTLEMENT | node_id | {"action_type": "BUILD_SETTLEMENT", "value": 23} |
| BUILD_CITY | node_id | {"action_type": "BUILD_CITY", "value": 23} |
| BUY_DEVELOPMENT_CARD | null | {"action_type": "BUY_DEVELOPMENT_CARD", "value": null} |
| PLAY_KNIGHT_CARD | null | {"action_type": "PLAY_KNIGHT_CARD", "value": null} |
| MOVE_ROBBER | [coord, victim] | {"action_type": "MOVE_ROBBER", "value": [[1,0,-1], "RED"]} |

### Trading with the Bank (MARITIME_TRADE):

Format: 5-element array [give1, give2, give3, give4, receive]
- **4:1 Bank Trade**: Give 4 of same resource, get 1 different
  `{"action_type": "MARITIME_TRADE", "value": ["WOOD","WOOD","WOOD","WOOD","ORE"]}`
  → Give 4 WOOD to bank, receive 1 ORE

- **3:1 Port Trade**: Give 3 of any resource (if you have a 3:1 port)
  `{"action_type": "MARITIME_TRADE", "value": ["BRICK","BRICK","BRICK",null,"WHEAT"]}`
  → Give 3 BRICK, receive 1 WHEAT

- **2:1 Port Trade**: Give 2 of specific resource (if you have that 2:1 port)
  `{"action_type": "MARITIME_TRADE", "value": ["ORE","ORE",null,null,"WOOD"]}`
  → Give 2 ORE, receive 1 WOOD

### Trading with Players (OFFER_TRADE):

Format: 10-element array [offer 5 resources, request 5 resources]
Order: [WOOD, BRICK, SHEEP, WHEAT, ORE, WOOD, BRICK, SHEEP, WHEAT, ORE]
         ^^^^^^^ YOU GIVE ^^^^^^^    ^^^^^^^ YOU GET ^^^^^^^

Example: Offer 1 wood + 1 brick, request 1 wheat
`{"action_type": "OFFER_TRADE", "value": [1,1,0,0,0, 0,0,0,1,0]}`

Note: Dice rolling is automatic. Choose wisely to reach 10 VP first!"""

    def get_output_schema(self) -> Dict[str, Any]:
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

    def format_valid_actions_for_prompt(
        self,
        playable_actions: List[Any],
    ) -> List[Dict[str, Any]]:
        """
        Convert Catanatron Action objects to prompt-friendly format.

        Args:
            playable_actions: List of Catanatron Action namedtuples

        Returns:
            List of dicts with action_type and value
        """
        formatted = []
        for action in playable_actions:
            formatted.append({
                "action_type": action.action_type.value,
                "value": self._format_action_value(action),
            })
        return formatted

    def _format_action_value(self, action: Any) -> Any:
        """Format action value for prompt display."""
        value = action.value
        if value is None:
            return None
        if isinstance(value, tuple):
            return list(value)
        return value
