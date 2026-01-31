# Game Interface

Every game in AGI-Evals implements four components that the platform orchestrates automatically. This document defines the contracts each component must satisfy.

## Directory Structure

```
games/<game_type>/
├── __init__.py          # Public exports + create_game() factory
├── game.py              # Game logic (implements core.Game)
├── state_adapter.py     # State → LLM prompt (implements core.StateAdapter)
├── action_parser.py     # LLM output → action (implements core.ActionParser)
└── config.py            # Pydantic config model
```

---

## 1. Game — `core.Game`

The central lifecycle interface. The arena calls these methods in a loop:

```
reset(seed)
  → loop:
      get_current_player()
      get_available_actions()
      get_public_state() / get_private_state(player_id)
      [agent decides]
      step(action)
  → is_over()
```

### Required Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `game_type` | `str` | Unique identifier (e.g., `"simple_card"`) |
| `reset(seed)` | `None` | Initialize or reinitialize all game state |
| `get_current_player()` | `str` | ID of the player whose turn it is. Return `""` when the game is over. |
| `get_current_role()` | `str` | Role of the current player (e.g., `"player"`, `"spymaster"`, `"operative"`) |
| `get_public_state()` | `dict` | State visible to all players (scores, board, history) |
| `get_private_state(player_id)` | `dict` | State visible only to that player (hand, secret info) |
| `get_available_actions()` | `list[dict]` | Legal actions for the current player. Each action must contain an `action_type` key. |
| `step(action)` | `(dict, bool)` | Execute an action. Return `(result_info, game_over)`. Raise `ValueError` for illegal actions. |
| `is_over()` | `bool` | Whether the game has ended |
| `get_winner()` | `str \| None` | Winner's player ID, or `None` if draw or still in progress |
| `get_scores()` | `dict` | Current scores for all players |

### Optional Methods

| Method | Default | Description |
|--------|---------|-------------|
| `get_players()` | `[]` | List of all player IDs |
| `get_teams()` | `None` | Team → player-ID mapping for team-based games |
| `serialize()` | — | Full state snapshot for logging and replay |

### Action Format

Each action is a `dict` with at minimum an `action_type` field:

```python
{"action_type": "PLAY_CARD", "card": 7}
```

`get_available_actions()` defines exactly which actions are legal. `step()` must raise `ValueError` if an invalid action is passed.

---

## 2. StateAdapter — `core.StateAdapter`

Converts structured game state into text prompts that an LLM can reason about.

### Required Methods

| Method | Description |
|--------|-------------|
| `state_to_prompt(public_state, private_state, valid_actions, turn_history)` | Main prompt body shown to the LLM |
| `get_output_schema()` | JSON Schema describing the expected LLM response format |

### Optional Overrides

| Method | Description |
|--------|-------------|
| `format_system_prompt(role, **kwargs)` | System-level prompt setting the LLM's persona. Receives `public_state` and `private_state` via kwargs so game-specific adapters can extract context. |
| `format_valid_actions_for_prompt(valid_actions)` | Reformat the action list for readability |
| `format_turn_history(turn_history, max_turns)` | Format recent history for inclusion in the prompt |

---

## 3. ActionParser — `core.ActionParser`

Extracts a valid game action from raw LLM output text. The runner calls this after the agent returns, making it the authoritative parser for each game.

### Required Methods

| Method | Description |
|--------|-------------|
| `parse(raw_output, valid_actions)` | Return a valid action dict from `valid_actions`, or raise `ActionParseError` |

### Provided Utilities

The base class provides these helper methods:

- `extract_json(raw_output)` — Finds a JSON object in code blocks or bare text
- `extract_reasoning(raw_output)` — Extracts text before the JSON block
- `find_closest_action(parsed, valid_actions)` — Matches by `action_type`
- `get_random_action(valid_actions)` — Returns a random valid action

Set `fallback_to_random=True` (the default) to return a random valid action on parse failure instead of raising.

---

## 4. Config

A Pydantic `BaseModel` defining game-specific parameters with defaults and validation. Registered as `config_class` in the registry and validated at arena startup.

```python
class MyGameConfig(BaseModel):
    board_size: int = Field(default=8, ge=2, le=20)

    class Config:
        extra = "forbid"
```

---

## 5. Factory Function

Each game module exports a `create_game(arena_config)` function — the single entry point the platform uses to instantiate the game:

```python
def create_game(arena_config):
    """Create a game instance from an ArenaConfig."""
    players = [p.id for p in arena_config.players]
    return MyGame(players=players, seed=arena_config.seed)
```

The factory receives the full `ArenaConfig` so it can extract player assignments, team structures, or any other setup logic your game requires.

---

## 6. Registration

Add your game to `GAME_REGISTRY` in `arena/registry.py`. All paths are resolved lazily at runtime via `importlib`, so the game module is only loaded when actually used:

```python
GAME_REGISTRY["my_game"] = {
    "game_module": "games.my_game",
    "game_class": "games.my_game.game.MyGame",
    "game_factory": "games.my_game.create_game",
    "state_adapter_class": "games.my_game.state_adapter.MyStateAdapter",
    "action_parser_class": "games.my_game.action_parser.MyActionParser",
    "config_class": "games.my_game.config.MyGameConfig",
}
```

---

## Example: `simple_card`

The `games/simple_card/` directory is a minimal reference implementation covering all required components. Study it as a starting template for new games.
