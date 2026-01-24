#!/usr/bin/env python3
"""
Demo: Single turn showing the full LLM prompt/response flow.

This script demonstrates what happens during one game turn:
1. Game state is converted to an LLM prompt
2. LLM responds with a JSON action
3. Response is parsed and validated
4. Action is executed

Run: python examples/demo_turn.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from catanatron.game import Game
from catanatron.models.player import RandomPlayer, Color
from catan_arena.adapters.catan_state_adapter import CatanStateAdapter
from catan_arena.adapters.catan_action_parser import CatanActionParser


def main():
    print("="*70)
    print("CATAN ARENA - SINGLE TURN DEMO")
    print("="*70)

    # 1. Create game
    print("\n[1] Creating game with 2 players...")
    players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
    game = Game(players)
    print(f"    Current player: {game.state.current_player().color.name}")
    print(f"    Available actions: {len(game.playable_actions)}")

    # 2. Create adapters
    adapter = CatanStateAdapter()
    parser = CatanActionParser()

    # 3. Build game state
    state = game.state
    board = state.board

    public_state = {
        "turn_number": 1,
        "current_player": state.current_player().color.name,
        "phase": "INITIAL_BUILD" if state.is_initial_build_phase else "MAIN",
        "robber": board.robber_coordinate,
        "players": {},
    }

    private_state = {
        "hand": {"WOOD": 0, "BRICK": 0, "SHEEP": 0, "WHEAT": 0, "ORE": 0},
    }

    # Limit to first 15 valid actions for readability
    valid_actions = [
        {"action_type": a.action_type.name, "value": a.value}
        for a in game.playable_actions[:15]
    ]

    # 4. Generate prompt
    print("\n[2] Generating LLM prompt...")
    prompt = adapter.state_to_prompt(
        public_state=public_state,
        private_state=private_state,
        valid_actions=valid_actions,
        turn_history=[],
    )

    print("\n" + "-"*70)
    print("PROMPT TO LLM:")
    print("-"*70)
    # Show truncated prompt
    lines = prompt.split('\n')
    for line in lines[:50]:
        print(line)
    if len(lines) > 50:
        print(f"... ({len(lines) - 50} more lines)")
    print("-"*70)

    # 5. Simulate LLM response
    # Pick a valid action to demonstrate
    first_action = game.playable_actions[0]
    simulated_response = f'''
Let me analyze the current game state.

## Analysis

This is the initial building phase, so I need to place my first settlement.
Looking at the available settlement locations, I should choose a spot that:
1. Has good resource diversity
2. Is near high-probability numbers (6, 8, 5, 9)
3. Allows for expansion

Node {first_action.value} looks like a strong position.

```json
{{
  "action_type": "{first_action.action_type.name}",
  "value": {first_action.value},
  "rationale": "Starting settlement at a strategic location with good resource access"
}}
```
'''

    print("\n[3] Simulated LLM response:")
    print("-"*70)
    print(simulated_response)
    print("-"*70)

    # 6. Parse response
    print("\n[4] Parsing LLM response...")
    try:
        parsed = parser.parse(simulated_response, valid_actions)
        print(f"    Parsed action: {parsed.get('action_type')}")
        print(f"    Value: {parsed.get('value')}")

        # 7. Execute action
        print("\n[5] Executing action...")
        game.execute(first_action)
        print(f"    Action executed: {first_action}")
        print(f"    Next player: {game.state.current_player().color.name}")
        print(f"    New available actions: {len(game.playable_actions)}")
    except Exception as e:
        print(f"    Parse error: {e}")

    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nTo run a full interactive game with manual LLM input:")
    print("    python examples/manual_play.py")


if __name__ == "__main__":
    main()
