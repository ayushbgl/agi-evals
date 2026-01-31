# Adding a New Game

This guide walks through adding a new game to AGI-Evals. No changes to the core engine or arena orchestration are needed — only files inside your new `games/<name>/` directory and a single registry entry.

## 1. Create the Game Directory

```bash
mkdir games/my_game
```

## 2. Implement the Game Logic

Create `games/my_game/game.py` with a class that extends `core.Game`:

```python
from typing import Any, Dict, List, Optional, Tuple
from core.game import Game

class MyGame(Game):
    def __init__(self, players: List[str], seed: Optional[int] = None):
        self._players = players
        self.reset(seed)

    @property
    def game_type(self) -> str:
        return "my_game"

    def reset(self, seed: Optional[int] = None) -> None:
        # Initialize all mutable game state here.
        ...

    def get_current_player(self) -> str:
        # Return the ID of whoever's turn it is.
        # Return "" when the game is over.
        ...

    def get_current_role(self) -> str:
        return "player"  # Or role-based if your game has distinct roles

    def get_public_state(self) -> Dict[str, Any]:
        # Everything visible to all players (scores, board, history).
        ...

    def get_private_state(self, player_id: str) -> Dict[str, Any]:
        # What only this player can see (e.g., their hand).
        ...

    def get_available_actions(self) -> List[Dict[str, Any]]:
        # Legal actions for the current player.
        # Each action must include an "action_type" key.
        ...

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        # Execute the action.
        # Return (result_info_dict, game_over_bool).
        # Raise ValueError for illegal actions.
        ...

    def is_over(self) -> bool:
        ...

    def get_winner(self) -> Optional[str]:
        # Return the winning player ID, or None for draw/ongoing.
        ...

    def get_scores(self) -> Dict[str, Any]:
        ...
```

## 3. Write the State Adapter

Create `games/my_game/state_adapter.py`. The adapter's job is to turn the structured game state into a prompt the LLM can reason about clearly:

```python
from typing import Any, Dict, List, Optional
from core.state_adapter import StateAdapter

class MyStateAdapter(StateAdapter):
    def state_to_prompt(
        self,
        public_state: Dict[str, Any],
        private_state: Dict[str, Any],
        valid_actions: List[Dict[str, Any]],
        turn_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        # Build a clear prompt covering:
        #   - Current game state (board, scores, etc.)
        #   - The player's private information (hand, role secrets)
        #   - Available actions with format instructions
        ...

    def get_output_schema(self) -> Dict[str, Any]:
        # JSON Schema for the expected LLM response
        return {
            "type": "object",
            "required": ["action_type"],
            "properties": {
                "action_type": {"type": "string"},
                "reasoning": {"type": "string"},
            }
        }
```

Override `format_system_prompt(role, **kwargs)` if your game needs a custom system prompt. The runner passes `public_state` and `private_state` via kwargs so you can extract game-specific context.

## 4. Write the Action Parser

Create `games/my_game/action_parser.py`. This is the authoritative parser — the runner calls it after the LLM responds to extract a valid action:

```python
from typing import Any, Dict, List
from core.action_parser import ActionParser, ActionParseError

class MyActionParser(ActionParser):
    def parse(
        self,
        raw_output: str,
        valid_actions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        # 1. Try JSON extraction (use self.extract_json())
        parsed = self.extract_json(raw_output)

        # 2. Validate/normalize against valid_actions
        if parsed:
            # Match parsed output to a legal action...
            ...

        # 3. Fall back to random on failure (if fallback_to_random is set)
        if self.fallback_to_random:
            return self.get_random_action(valid_actions)
        raise ActionParseError("Could not parse output", raw_output)
```

## 5. Add Configuration

Create `games/my_game/config.py` with game-specific parameters:

```python
from pydantic import BaseModel, Field

class MyGameConfig(BaseModel):
    board_size: int = Field(default=8, ge=2, le=20)

    class Config:
        extra = "forbid"
```

## 6. Wire Up the Module

Create `games/my_game/__init__.py` with exports and the factory function:

```python
from games.my_game.game import MyGame
from games.my_game.config import MyGameConfig
from games.my_game.state_adapter import MyStateAdapter
from games.my_game.action_parser import MyActionParser

GAME_TYPE = "my_game"

def create_game(arena_config):
    """Factory: create a MyGame from ArenaConfig."""
    players = [p.id for p in arena_config.players]
    return MyGame(players=players, seed=arena_config.seed)
```

## 7. Register the Game

Add one entry to `GAME_REGISTRY` in `arena/registry.py`:

```python
"my_game": {
    "game_module": "games.my_game",
    "game_class": "games.my_game.game.MyGame",
    "game_factory": "games.my_game.create_game",
    "state_adapter_class": "games.my_game.state_adapter.MyStateAdapter",
    "action_parser_class": "games.my_game.action_parser.MyActionParser",
    "config_class": "games.my_game.config.MyGameConfig",
},
```

## 8. Verify with Random Agents

Run a smoke test with random agents to confirm the full loop works:

```python
from arena.orchestration.runner import run_simple_game

result = run_simple_game(
    game_type="my_game",
    player_configs=[
        {"id": "player_0", "type": "random"},
        {"id": "player_1", "type": "random"},
    ],
    max_turns=100,
    verbose=True,
)
print(f"Winner: {result['winner']}")
print(f"Scores: {result['final_scores']}")
```

## Checklist

- [ ] `games/my_game/game.py` — Game class extending `core.Game`
- [ ] `games/my_game/state_adapter.py` — StateAdapter for LLM prompts
- [ ] `games/my_game/action_parser.py` — ActionParser for LLM output
- [ ] `games/my_game/config.py` — Pydantic config model
- [ ] `games/my_game/__init__.py` — Exports and `create_game()` factory
- [ ] `arena/registry.py` — Entry added to `GAME_REGISTRY`
- [ ] Smoke test passes with random agents

## Reference Implementation

The `games/simple_card/` directory is a minimal, fully-working example covering all required components. Use it as a template.
