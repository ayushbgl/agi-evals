#!/usr/bin/env python3
"""
Manual Play Mode - Test LLMs by copy-pasting prompts

This script runs a Catan game where you manually relay messages
between the game and LLM UIs (Claude, ChatGPT, Gemini, etc).

Usage:
    python examples/manual_play.py

For each LLM turn:
1. The script shows you the prompt to copy
2. You paste it into your LLM UI
3. You copy the LLM's response back here
4. The game continues

Notes:
- ROLL actions are automatic (no LLM decision needed)
- LLMs are informed of dice results and recent game actions
- Initial placement phase has no dice rolling
"""

import sys
import json
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from catanatron.game import Game
from catanatron.models.player import Player, RandomPlayer, Color
from catanatron.models.enums import ActionType

from catan_arena.adapters.catan_state_adapter import CatanStateAdapter
from catan_arena.adapters.catan_action_parser import CatanActionParser
from catan_arena.adapters.board_renderer import render_board_compact, get_node_resource_info


# Shared game history across all players
GAME_HISTORY = []
LAST_DICE_ROLL = None


class LLMForfeitError(Exception):
    """Raised when an LLM fails to provide a valid action after max attempts."""
    def __init__(self, llm_name: str, color: str):
        self.llm_name = llm_name
        self.color = color
        super().__init__(f"{llm_name} ({color}) forfeited due to invalid responses")


class ManualLLMPlayer(Player):
    """A player that prompts for manual LLM input."""

    def __init__(self, color: Color, name: str = "ManualLLM"):
        super().__init__(color)
        self.name = name
        self.state_adapter = CatanStateAdapter()
        # Disable fallback_to_random - we want to catch invalid actions!
        self.action_parser = CatanActionParser(fallback_to_random=False)

    def decide(self, game: Game, playable_actions):
        """Show prompt and get manual input. LLM gets 3 attempts before forfeit."""
        global GAME_HISTORY, LAST_DICE_ROLL

        actions_list = list(playable_actions)
        max_attempts = 3

        # Detect context: regular turn, trade response, or trade confirmation
        action_types = set(a.action_type.name for a in actions_list)
        is_trade_response = "ACCEPT_TRADE" in action_types or "REJECT_TRADE" in action_types
        is_trade_confirm = "CONFIRM_TRADE" in action_types or "CANCEL_TRADE" in action_types

        # Build the observation
        public_state = self._extract_public_state(game)
        private_state = self._extract_private_state(game)
        valid_actions = self._format_actions(actions_list)

        # Add dice roll info if available
        if LAST_DICE_ROLL and not game.state.is_initial_build_phase:
            public_state["last_dice_roll"] = {
                "dice": LAST_DICE_ROLL,
                "total": sum(LAST_DICE_ROLL),
            }

        # Add trade context to public state if in trade mode
        if is_trade_response and game.state.current_trade:
            trade = game.state.current_trade
            public_state["pending_trade"] = {
                "offering": {"WOOD": trade[0], "BRICK": trade[1], "SHEEP": trade[2], "WHEAT": trade[3], "ORE": trade[4]},
                "requesting": {"WOOD": trade[5], "BRICK": trade[6], "SHEEP": trade[7], "WHEAT": trade[8], "ORE": trade[9]},
                "from_player_index": trade[10] if len(trade) > 10 else None,
            }

        # Generate board rendering
        board_ascii = render_board_compact(game)
        node_resources = get_node_resource_info(game)

        # Generate prompt with shared game history
        prompt = self.state_adapter.state_to_prompt(
            public_state=public_state,
            private_state=private_state,
            valid_actions=valid_actions,
            turn_history=GAME_HISTORY[-15:],  # Last 15 actions from all players
        )

        # Prepend board state to prompt
        prompt = board_ascii + "\n\n" + prompt

        # Determine turn type for display
        if is_trade_response:
            turn_type = "TRADE RESPONSE"
        elif is_trade_confirm:
            turn_type = "CONFIRM TRADE"
        else:
            turn_type = "TURN"

        for attempt in range(1, max_attempts + 1):
            # Display prompt for user to copy
            print("\n" + "="*70)
            print(f"🤖 {self.name}'s {turn_type} ({self.color.name}) - Attempt {attempt}/{max_attempts}")
            print("="*70)

            if attempt == 1:
                print("\n📋 COPY THIS PROMPT TO YOUR LLM:\n")
                print("-"*70)
                print(prompt)
                print("-"*70)
            else:
                print(f"\n⚠️  Previous response was invalid. {max_attempts - attempt + 1} attempts remaining.")
                print(f"Valid action types: {set(a.action_type.name for a in actions_list)}")
                print("\nPaste the corrected LLM response:")

            # Get response from user
            print(f"\n📝 PASTE {self.name}'s RESPONSE (end with empty line):\n")
            lines = []
            while True:
                try:
                    line = input()
                    if line == "":
                        if lines:  # Only break if we have some content
                            break
                    lines.append(line)
                except EOFError:
                    break

            response = "\n".join(lines)

            # Parse the response and validate against valid actions
            try:
                parsed = self.action_parser.parse(response, valid_actions)
                action_type = parsed.get("action_type")
                value = parsed.get("value")

                # Check if action type is valid for this turn
                valid_types = set(a.action_type.name for a in actions_list)
                if action_type not in valid_types:
                    print(f"\n❌ INVALID: '{action_type}' is not valid right now.")
                    print(f"   Valid action types: {', '.join(sorted(valid_types))}")
                    continue  # Try again

                # Find matching action with exact value
                for action in actions_list:
                    if action.action_type.name == action_type:
                        if self._action_matches(action, value):
                            print(f"\n✅ Action: {action}")
                            # Record to shared game history
                            GAME_HISTORY.append({
                                "player": self.color.name,
                                "action_type": action_type,
                                "value": value,
                            })
                            return action

                # Action type is valid but value doesn't match any option
                valid_values = [a.value for a in actions_list if a.action_type.name == action_type]
                print(f"\n❌ INVALID: value '{value}' is not valid for {action_type}.")
                print(f"   Valid values: {valid_values}")
                continue  # Try again

            except Exception as e:
                print(f"\n⚠️  Could not parse LLM response: {e}")
                continue  # Try again

        # All attempts exhausted - LLM loses!
        print("\n" + "="*70)
        print(f"💀 {self.name} ({self.color.name}) FORFEITS!")
        print(f"   Failed to provide a valid action after {max_attempts} attempts.")
        print("="*70)
        raise LLMForfeitError(self.name, self.color.name)

    def _action_matches(self, action, value):
        """Check if action matches the parsed value."""
        if value is None:
            return True

        # Handle different action types
        if action.action_type == ActionType.BUILD_SETTLEMENT:
            return action.value == value or str(action.value) == str(value)
        elif action.action_type == ActionType.BUILD_ROAD:
            if isinstance(value, (list, tuple)) and len(value) == 2:
                edge = tuple(sorted([value[0], value[1]]))
                action_edge = tuple(sorted(action.value))
                return edge == action_edge
        elif action.action_type == ActionType.MOVE_ROBBER:
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                return tuple(value[:3]) == tuple(action.value[:3])

        return str(action.value) == str(value)

    def _extract_public_state(self, game):
        """Extract public game state."""
        state = game.state
        board = state.board

        return {
            "turn_number": game.state.num_turns,
            "current_player": self.color.name,
            "phase": "INITIAL_BUILD" if state.is_initial_build_phase else "MAIN",
            "players": {
                p.color.name: {
                    "victory_points": state.player_state[f"P{i}_VICTORY_POINTS"],
                    "resource_count": sum(
                        state.player_state.get(f"P{i}_{r}_IN_HAND", 0)
                        for r in ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
                    ),
                    "dev_cards": state.player_state.get(f"P{i}_NUM_DEVCARD", 0),
                    "knights_played": state.player_state.get(f"P{i}_PLAYED_KNIGHT", 0),
                }
                for i, p in enumerate(game.state.players)
            },
            "robber": board.robber_coordinate,
            "buildings": len(board.buildings),
            "roads": len(board.roads),
        }

    def _extract_private_state(self, game):
        """Extract private state for this player."""
        state = game.state
        player_idx = next(
            i for i, p in enumerate(state.players) if p.color == self.color
        )
        prefix = f"P{player_idx}_"

        return {
            "hand": {
                "WOOD": state.player_state.get(f"{prefix}WOOD_IN_HAND", 0),
                "BRICK": state.player_state.get(f"{prefix}BRICK_IN_HAND", 0),
                "SHEEP": state.player_state.get(f"{prefix}SHEEP_IN_HAND", 0),
                "WHEAT": state.player_state.get(f"{prefix}WHEAT_IN_HAND", 0),
                "ORE": state.player_state.get(f"{prefix}ORE_IN_HAND", 0),
            },
            "dev_cards": {
                "KNIGHT": state.player_state.get(f"{prefix}KNIGHT_IN_HAND", 0),
                "VICTORY_POINT": state.player_state.get(f"{prefix}VICTORY_POINT_IN_HAND", 0),
                "ROAD_BUILDING": state.player_state.get(f"{prefix}ROAD_BUILDING_IN_HAND", 0),
                "YEAR_OF_PLENTY": state.player_state.get(f"{prefix}YEAR_OF_PLENTY_IN_HAND", 0),
                "MONOPOLY": state.player_state.get(f"{prefix}MONOPOLY_IN_HAND", 0),
            },
        }

    def _format_actions(self, actions):
        """Format actions for the prompt."""
        formatted = []
        for action in actions:
            formatted.append({
                "action_type": action.action_type.name,
                "value": action.value if action.value is not None else None,
            })
        return formatted


def setup_game():
    """Setup game with player configuration."""
    print("\n" + "="*70)
    print("🎲 CATAN ARENA - MANUAL LLM TESTING MODE")
    print("="*70)
    print("\nThis mode lets you test LLMs by manually copy-pasting prompts.")
    print("You can use any LLM UI (Claude, ChatGPT, Gemini, etc.)\n")

    # Get number of players
    print("How many players? (2-4)")
    while True:
        try:
            num_players = int(input("Number of players [2]: ").strip() or "2")
            if 2 <= num_players <= 4:
                break
            print("Please enter 2-4")
        except ValueError:
            print("Please enter a number")

    colors = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE][:num_players]
    players = []

    print(f"\nFor each player, choose type:")
    print("  [m] Manual LLM (you copy-paste prompts)")
    print("  [r] Random bot")
    print()

    for i, color in enumerate(colors):
        while True:
            choice = input(f"Player {i+1} ({color.name}) [m/r]: ").strip().lower() or "m"
            if choice == "m":
                name = input(f"  LLM name (e.g., 'Claude', 'GPT-4'): ").strip() or f"LLM_{color.name}"
                players.append(ManualLLMPlayer(color, name))
                break
            elif choice == "r":
                players.append(RandomPlayer(color))
                break
            print("Please enter 'm' or 'r'")

    return players


def run_game(players):
    """Run the game loop."""
    global GAME_HISTORY, LAST_DICE_ROLL

    # Reset shared state
    GAME_HISTORY = []
    LAST_DICE_ROLL = None

    game = Game(players)

    print("\n" + "="*70)
    print("🎮 GAME STARTED!")
    print("="*70)
    print(f"\nPlayers: {[f'{p.color.name} ({type(p).__name__})' for p in players]}")
    print("\nThe game will now begin. For Manual LLM players,")
    print("copy the prompt to your LLM and paste back the response.")
    print("\nNote: Dice rolls are automatic. Initial placement has no rolling.\n")

    turn_count = 0
    max_turns = 500
    forfeit_loser = None

    try:
        while game.winning_color() is None and turn_count < max_turns:
            current_player = game.state.current_player()
            playable_actions = game.playable_actions

            if not playable_actions:
                print("No playable actions - this shouldn't happen!")
                break

            # Auto-execute ROLL - dice rolling is automatic, no decision needed
            roll_actions = [a for a in playable_actions if a.action_type == ActionType.ROLL]
            if roll_actions:
                action = roll_actions[0]
                action_record = game.execute(action)
                # Capture dice result
                if action_record and action_record.result:
                    LAST_DICE_ROLL = action_record.result
                    dice_total = sum(LAST_DICE_ROLL)
                    print(f"  🎲 {current_player.color.name} rolled {LAST_DICE_ROLL[0]} + {LAST_DICE_ROLL[1]} = {dice_total}")
                    # Record to history
                    GAME_HISTORY.append({
                        "player": current_player.color.name,
                        "action_type": "ROLL",
                        "value": list(LAST_DICE_ROLL),
                        "total": dice_total,
                    })
                turn_count += 1
                continue

            # Check what kind of action context we're in
            action_types = set(a.action_type.name for a in playable_actions)
            is_trade_response = "ACCEPT_TRADE" in action_types
            is_trade_confirm = "CONFIRM_TRADE" in action_types

            # Get action from current player
            action = current_player.decide(game, playable_actions)

            # Execute action
            game.execute(action)

            # Show what happened and record to history
            action_name = action.action_type.name
            if isinstance(current_player, RandomPlayer):
                if action_name not in ["ROLL", "END_TURN"]:
                    print(f"  🤖 {current_player.color.name} (Random): {action_name} {action.value if action.value else ''}")
                    GAME_HISTORY.append({
                        "player": current_player.color.name,
                        "action_type": action_name,
                        "value": action.value,
                    })
                elif action_name == "END_TURN":
                    GAME_HISTORY.append({
                        "player": current_player.color.name,
                        "action_type": "END_TURN",
                    })
            else:
                # For LLM players, show trade responses
                if is_trade_response:
                    if action_name == "ACCEPT_TRADE":
                        print(f"  ✅ {current_player.color.name} ACCEPTED the trade")
                    else:
                        print(f"  ❌ {current_player.color.name} REJECTED the trade")
                elif is_trade_confirm:
                    if action_name == "CONFIRM_TRADE":
                        print(f"  🤝 {current_player.color.name} CONFIRMED trade")
                    else:
                        print(f"  🚫 {current_player.color.name} CANCELLED trade")
                elif action_name == "END_TURN":
                    print(f"  ⏭️  {current_player.color.name} ended turn")

            turn_count += 1

            # Periodic status
            if turn_count % 50 == 0:
                print(f"\n--- Turn {turn_count} ---")
                for p in players:
                    vp = game.state.player_state.get(
                        f"P{players.index(p)}_VICTORY_POINTS", 0
                    )
                    print(f"  {p.color.name}: {vp} VP")

    except LLMForfeitError as e:
        forfeit_loser = e.color
        print(f"\n{e.llm_name} loses by forfeit!")

    # Game over
    print("\n" + "="*70)
    print("🏆 GAME OVER!")
    print("="*70)

    winner = game.winning_color()
    if forfeit_loser:
        # Find the other players as potential winners
        other_players = [p for p in players if p.color.name != forfeit_loser]
        if len(other_players) == 1:
            winner_player = other_players[0]
            winner_name = getattr(winner_player, 'name', type(winner_player).__name__)
            print(f"\n🎉 Winner by forfeit: {winner_player.color.name} ({winner_name})")
        else:
            # Multiple players remain - highest VP wins
            scores = [(p, game.state.player_state.get(f"P{players.index(p)}_VICTORY_POINTS", 0))
                      for p in other_players]
            scores.sort(key=lambda x: -x[1])
            winner_player = scores[0][0]
            winner_name = getattr(winner_player, 'name', type(winner_player).__name__)
            print(f"\n🎉 Winner by forfeit (highest VP): {winner_player.color.name} ({winner_name})")
    elif winner:
        winner_player = next(p for p in players if p.color == winner)
        winner_name = getattr(winner_player, 'name', type(winner_player).__name__)
        print(f"\n🎉 Winner: {winner.name} ({winner_name})")
    else:
        print(f"\n⏰ Game ended after {turn_count} turns (no winner)")

    print("\nFinal Scores:")
    for i, p in enumerate(players):
        vp = game.state.player_state.get(f"P{i}_VICTORY_POINTS", 0)
        name = getattr(p, 'name', type(p).__name__)
        print(f"  {p.color.name} ({name}): {vp} VP")

    return game


def main():
    """Main entry point."""
    try:
        players = setup_game()
        game = run_game(players)

        # Ask to save log
        save = input("\nSave game log? [y/N]: ").strip().lower()
        if save == 'y':
            from datetime import datetime
            filename = f"manual_game_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            # Basic log saving would go here
            print(f"Game log saved to: {filename}")

    except KeyboardInterrupt:
        print("\n\nGame interrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
