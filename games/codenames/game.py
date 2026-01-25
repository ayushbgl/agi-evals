"""
Core Codenames game logic.

Implements the full Codenames rules:
- 5x5 grid of 25 words
- Two teams (Red and Blue) with Spymasters and Operatives
- Spymasters give one-word clues with a number
- Operatives guess words based on clues
- First team to find all their agents wins
- Touching the assassin causes instant loss
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
import random

from core.game import Game
from games.codenames.words import select_game_words, get_word_list


class Team(Enum):
    """Team identifiers."""
    RED = "red"
    BLUE = "blue"


class CardType(Enum):
    """Types of cards on the board."""
    RED = "red"
    BLUE = "blue"
    BYSTANDER = "bystander"
    ASSASSIN = "assassin"


class Phase(Enum):
    """Game phases within a turn."""
    SPYMASTER_CLUE = "spymaster_clue"
    OPERATIVE_GUESS = "operative_guess"
    GAME_OVER = "game_over"


@dataclass
class ClueRecord:
    """Record of a clue given by a spymaster."""
    team: Team
    word: str
    number: int
    turn_number: int


@dataclass
class GuessRecord:
    """Record of a guess made by an operative."""
    team: Team
    player_id: str
    word: str
    card_type: CardType
    correct: bool  # Was it their team's card?
    turn_number: int


@dataclass
class ClueResult:
    """Result of giving a clue."""
    valid: bool
    error: Optional[str] = None


@dataclass
class GuessResult:
    """Result of making a guess."""
    word: str
    card_type: CardType
    correct: bool  # Was it the guessing team's card?
    turn_continues: bool  # Does the team get to keep guessing?
    game_over: bool
    winner: Optional[Team] = None
    error: Optional[str] = None


class CodenamesGame(Game):
    """
    Full Codenames game implementation.

    Supports:
    - Standard 5x5 grid with 25 words
    - Red vs Blue teams
    - Spymaster gives clues, Operatives guess
    - Multiple operatives per team (sequential turns)
    - Full win/lose conditions including assassin
    """

    def __init__(
        self,
        red_spymaster: str,
        red_operatives: List[str],
        blue_spymaster: str,
        blue_operatives: List[str],
        word_list: Optional[List[str]] = None,
        starting_team: Optional[Team] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize a new Codenames game.

        Args:
            red_spymaster: Player ID for red team's spymaster
            red_operatives: Player IDs for red team's operatives
            blue_spymaster: Player ID for blue team's spymaster
            blue_operatives: Player IDs for blue team's operatives
            word_list: Custom word list (default: standard 400 words)
            starting_team: Which team goes first (random if None)
            seed: Random seed for reproducibility
        """
        self._seed = seed
        self._rng = random.Random(seed)

        # Store player assignments
        self.red_spymaster = red_spymaster
        self.red_operatives = list(red_operatives)
        self.blue_spymaster = blue_spymaster
        self.blue_operatives = list(blue_operatives)

        # Build player -> team/role mappings
        self._player_team: Dict[str, Team] = {}
        self._player_role: Dict[str, str] = {}

        self._player_team[red_spymaster] = Team.RED
        self._player_role[red_spymaster] = "spymaster"
        for op in red_operatives:
            self._player_team[op] = Team.RED
            self._player_role[op] = "operative"

        self._player_team[blue_spymaster] = Team.BLUE
        self._player_role[blue_spymaster] = "spymaster"
        for op in blue_operatives:
            self._player_team[op] = Team.BLUE
            self._player_role[op] = "operative"

        # Store word list
        self._word_list = word_list or get_word_list("standard")

        # Determine starting team
        if starting_team is None:
            starting_team = self._rng.choice([Team.RED, Team.BLUE])
        self._starting_team = starting_team

        # Initialize game state
        self.reset(seed)

    @property
    def game_type(self) -> str:
        return "codenames"

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the game to initial state."""
        if seed is not None:
            self._seed = seed
            self._rng = random.Random(seed)

        # Select 25 words for the grid
        words = select_game_words(self._word_list, 25, self._seed)

        # Arrange into 5x5 grid
        self.grid: List[List[str]] = [
            words[i*5:(i+1)*5] for i in range(5)
        ]

        # Flatten for easy lookup
        self._all_words: List[str] = words

        # Assign card types (key card)
        self.card_types: Dict[str, CardType] = self._generate_key_card()

        # Track revealed cards
        self.revealed: Set[str] = set()

        # Track remaining agents per team
        self.red_remaining = sum(1 for ct in self.card_types.values() if ct == CardType.RED)
        self.blue_remaining = sum(1 for ct in self.card_types.values() if ct == CardType.BLUE)

        # Turn state
        self.current_team: Team = self._starting_team
        self.current_phase: Phase = Phase.SPYMASTER_CLUE
        self.turn_number: int = 1

        # Current clue (set when spymaster gives clue)
        self.current_clue: Optional[Tuple[str, int]] = None
        self.guesses_remaining: int = 0

        # For multiple operatives: track whose turn it is
        self._current_operative_index: int = 0

        # History
        self.clue_history: List[ClueRecord] = []
        self.guess_history: List[GuessRecord] = []

        # Game end state
        self._winner: Optional[Team] = None
        self._game_over: bool = False
        self._loss_reason: Optional[str] = None

    def _generate_key_card(self) -> Dict[str, CardType]:
        """
        Generate random card type assignments.

        Starting team gets 9 cards, other team gets 8.
        7 bystanders and 1 assassin.
        """
        types = []

        # Starting team gets 9
        if self._starting_team == Team.RED:
            types.extend([CardType.RED] * 9)
            types.extend([CardType.BLUE] * 8)
        else:
            types.extend([CardType.BLUE] * 9)
            types.extend([CardType.RED] * 8)

        # Add bystanders and assassin
        types.extend([CardType.BYSTANDER] * 7)
        types.append(CardType.ASSASSIN)

        # Shuffle and assign
        self._rng.shuffle(types)

        return {word: card_type for word, card_type in zip(self._all_words, types)}

    def get_players(self) -> List[str]:
        """Get all player IDs."""
        return list(self._player_team.keys())

    def get_teams(self) -> Dict[str, List[str]]:
        """Get team assignments."""
        return {
            "red": [self.red_spymaster] + self.red_operatives,
            "blue": [self.blue_spymaster] + self.blue_operatives,
        }

    def get_state(self) -> Dict[str, Any]:
        """Get complete game state."""
        return {
            "public": self.get_public_state(),
            "private_states": {
                player_id: self.get_private_state(player_id)
                for player_id in self.get_players()
            },
            "metadata": {
                "turn_number": self.turn_number,
                "phase": self.current_phase.value,
                "current_team": self.current_team.value,
                "game_over": self._game_over,
                "winner": self._winner.value if self._winner else None,
            }
        }

    def get_public_state(self) -> Dict[str, Any]:
        """Get publicly visible state (what operatives can see)."""
        # Build grid with revealed status
        grid_state = []
        for row in self.grid:
            row_state = []
            for word in row:
                word_state = {
                    "word": word,
                    "revealed": word in self.revealed,
                }
                if word in self.revealed:
                    word_state["card_type"] = self.card_types[word].value
                row_state.append(word_state)
            grid_state.append(row_state)

        return {
            "grid": grid_state,
            "revealed_words": list(self.revealed),
            "current_team": self.current_team.value,
            "current_phase": self.current_phase.value,
            "current_clue": {
                "word": self.current_clue[0],
                "number": self.current_clue[1],
            } if self.current_clue else None,
            "guesses_remaining": self.guesses_remaining,
            "red_remaining": self.red_remaining,
            "blue_remaining": self.blue_remaining,
            "turn_number": self.turn_number,
            "clue_history": [
                {
                    "team": c.team.value,
                    "word": c.word,
                    "number": c.number,
                    "turn": c.turn_number,
                }
                for c in self.clue_history
            ],
            "guess_history": [
                {
                    "team": g.team.value,
                    "player": g.player_id,
                    "word": g.word,
                    "card_type": g.card_type.value,
                    "correct": g.correct,
                    "turn": g.turn_number,
                }
                for g in self.guess_history[-10:]  # Last 10 guesses
            ],
            "game_over": self._game_over,
            "winner": self._winner.value if self._winner else None,
        }

    def get_private_state(self, player_id: str) -> Dict[str, Any]:
        """Get private state for a specific player."""
        team = self._player_team.get(player_id)
        role = self._player_role.get(player_id)

        private_state = {
            "player_id": player_id,
            "team": team.value if team else None,
            "role": role,
        }

        # Spymasters can see all card types
        if role == "spymaster":
            private_state["card_types"] = {
                word: ct.value for word, ct in self.card_types.items()
            }
            private_state["key_card"] = self._format_key_card_for_display()

        return private_state

    def _format_key_card_for_display(self) -> List[List[str]]:
        """Format the key card as a 5x5 grid of card type initials."""
        key_card = []
        for row in self.grid:
            key_row = []
            for word in row:
                ct = self.card_types[word]
                if ct == CardType.RED:
                    key_row.append("R")
                elif ct == CardType.BLUE:
                    key_row.append("B")
                elif ct == CardType.BYSTANDER:
                    key_row.append("-")
                else:  # ASSASSIN
                    key_row.append("X")
            key_card.append(key_row)
        return key_card

    def get_current_player(self) -> str:
        """Get the ID of the current player."""
        if self.current_phase == Phase.SPYMASTER_CLUE:
            if self.current_team == Team.RED:
                return self.red_spymaster
            else:
                return self.blue_spymaster
        elif self.current_phase == Phase.OPERATIVE_GUESS:
            operatives = (
                self.red_operatives if self.current_team == Team.RED
                else self.blue_operatives
            )
            return operatives[self._current_operative_index % len(operatives)]
        else:
            # Game over - no current player
            return ""

    def get_current_role(self) -> str:
        """Get the current player's role."""
        if self.current_phase == Phase.SPYMASTER_CLUE:
            return "spymaster"
        elif self.current_phase == Phase.OPERATIVE_GUESS:
            return "operative"
        else:
            return ""

    def get_available_actions(self) -> List[Dict[str, Any]]:
        """Get valid actions for the current player."""
        if self._game_over:
            return []

        if self.current_phase == Phase.SPYMASTER_CLUE:
            # Spymaster can give any clue (word + number)
            # We don't enumerate all possible clues - just describe the action format
            return [{
                "action_type": "GIVE_CLUE",
                "description": "Give a one-word clue and a number",
                "format": {"clue": "string", "number": "int (0-9)"},
            }]

        elif self.current_phase == Phase.OPERATIVE_GUESS:
            # Operatives can guess any unrevealed word or pass
            actions = []

            # Add guess actions for each unrevealed word
            for word in self._all_words:
                if word not in self.revealed:
                    actions.append({
                        "action_type": "GUESS",
                        "word": word,
                    })

            # Add pass action
            actions.append({
                "action_type": "PASS",
                "description": "End guessing and pass to other team",
            })

            return actions

        return []

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """
        Execute an action and advance game state.

        Args:
            action: Action dict with action_type and parameters

        Returns:
            Tuple of (result_dict, game_over)
        """
        action_type = action.get("action_type")

        if action_type == "GIVE_CLUE":
            clue_word = action.get("clue", "").strip().upper()
            clue_number = action.get("number", 0)
            result = self.give_clue(clue_word, clue_number)
            return {
                "action": "GIVE_CLUE",
                "clue": clue_word,
                "number": clue_number,
                "valid": result.valid,
                "error": result.error,
            }, self._game_over

        elif action_type == "GUESS":
            word = action.get("word", "").strip().upper()
            result = self.make_guess(word)
            return {
                "action": "GUESS",
                "word": word,
                "card_type": result.card_type.value if result.card_type else None,
                "correct": result.correct,
                "turn_continues": result.turn_continues,
                "game_over": result.game_over,
                "winner": result.winner.value if result.winner else None,
                "error": result.error,
            }, self._game_over

        elif action_type == "PASS":
            self._end_turn()
            return {
                "action": "PASS",
                "next_team": self.current_team.value,
            }, self._game_over

        else:
            return {
                "error": f"Unknown action type: {action_type}",
            }, self._game_over

    def give_clue(self, word: str, number: int) -> ClueResult:
        """
        Spymaster gives a clue.

        Args:
            word: One-word clue
            number: Number of related words (0-9, or unlimited)

        Returns:
            ClueResult indicating success or error
        """
        if self.current_phase != Phase.SPYMASTER_CLUE:
            return ClueResult(valid=False, error="Not spymaster clue phase")

        # Validate clue word
        word = word.strip().upper()

        # Check if clue is one of the words on the board (not allowed)
        if word in self._all_words:
            return ClueResult(valid=False, error=f"Clue cannot be a word on the board: {word}")

        # Check number is valid
        if not isinstance(number, int) or number < 0:
            return ClueResult(valid=False, error="Number must be a non-negative integer")

        # Record the clue
        self.current_clue = (word, number)
        self.guesses_remaining = number + 1 if number > 0 else 100  # Unlimited if 0

        self.clue_history.append(ClueRecord(
            team=self.current_team,
            word=word,
            number=number,
            turn_number=self.turn_number,
        ))

        # Move to operative guess phase
        self.current_phase = Phase.OPERATIVE_GUESS
        self._current_operative_index = 0

        return ClueResult(valid=True)

    def make_guess(self, word: str) -> GuessResult:
        """
        Operative makes a guess.

        Args:
            word: The word to guess

        Returns:
            GuessResult with outcome
        """
        if self.current_phase != Phase.OPERATIVE_GUESS:
            return GuessResult(
                word=word,
                card_type=CardType.BYSTANDER,
                correct=False,
                turn_continues=False,
                game_over=False,
                error="Not operative guess phase"
            )

        word = word.strip().upper()

        # Validate word exists and not revealed
        if word not in self._all_words:
            return GuessResult(
                word=word,
                card_type=CardType.BYSTANDER,
                correct=False,
                turn_continues=False,
                game_over=False,
                error=f"Word not on board: {word}"
            )

        if word in self.revealed:
            return GuessResult(
                word=word,
                card_type=self.card_types[word],
                correct=False,
                turn_continues=False,
                game_over=False,
                error=f"Word already revealed: {word}"
            )

        # Reveal the card
        self.revealed.add(word)
        card_type = self.card_types[word]
        current_player = self.get_current_player()

        # Update remaining counts
        if card_type == CardType.RED:
            self.red_remaining -= 1
        elif card_type == CardType.BLUE:
            self.blue_remaining -= 1

        # Determine outcome
        is_team_card = (
            (self.current_team == Team.RED and card_type == CardType.RED) or
            (self.current_team == Team.BLUE and card_type == CardType.BLUE)
        )

        # Record the guess
        self.guess_history.append(GuessRecord(
            team=self.current_team,
            player_id=current_player,
            word=word,
            card_type=card_type,
            correct=is_team_card,
            turn_number=self.turn_number,
        ))

        # Check for assassin (instant loss)
        if card_type == CardType.ASSASSIN:
            self._game_over = True
            self._winner = Team.BLUE if self.current_team == Team.RED else Team.RED
            self._loss_reason = "assassin"
            self.current_phase = Phase.GAME_OVER
            return GuessResult(
                word=word,
                card_type=card_type,
                correct=False,
                turn_continues=False,
                game_over=True,
                winner=self._winner,
            )

        # Check for win (all team's agents found)
        if self.red_remaining == 0:
            self._game_over = True
            self._winner = Team.RED
            self.current_phase = Phase.GAME_OVER
            return GuessResult(
                word=word,
                card_type=card_type,
                correct=is_team_card,
                turn_continues=False,
                game_over=True,
                winner=Team.RED,
            )

        if self.blue_remaining == 0:
            self._game_over = True
            self._winner = Team.BLUE
            self.current_phase = Phase.GAME_OVER
            return GuessResult(
                word=word,
                card_type=card_type,
                correct=is_team_card,
                turn_continues=False,
                game_over=True,
                winner=Team.BLUE,
            )

        # Determine if turn continues
        self.guesses_remaining -= 1

        if is_team_card and self.guesses_remaining > 0:
            # Correct guess, can continue
            # Rotate to next operative
            operatives = (
                self.red_operatives if self.current_team == Team.RED
                else self.blue_operatives
            )
            self._current_operative_index = (
                self._current_operative_index + 1
            ) % len(operatives)

            return GuessResult(
                word=word,
                card_type=card_type,
                correct=True,
                turn_continues=True,
                game_over=False,
            )
        else:
            # Wrong guess or out of guesses - turn ends
            self._end_turn()
            return GuessResult(
                word=word,
                card_type=card_type,
                correct=is_team_card,
                turn_continues=False,
                game_over=False,
            )

    def _end_turn(self) -> None:
        """End current team's turn and switch to other team."""
        # Switch teams
        self.current_team = Team.BLUE if self.current_team == Team.RED else Team.RED

        # Reset for new turn
        self.current_phase = Phase.SPYMASTER_CLUE
        self.current_clue = None
        self.guesses_remaining = 0
        self._current_operative_index = 0
        self.turn_number += 1

    def is_over(self) -> bool:
        """Check if game has ended."""
        return self._game_over

    def get_winner(self) -> Optional[str]:
        """Get the winning team (as string) or None."""
        return self._winner.value if self._winner else None

    def get_scores(self) -> Dict[str, Any]:
        """Get current scores (agents found)."""
        # Count revealed agents per team
        red_found = sum(
            1 for word in self.revealed
            if self.card_types[word] == CardType.RED
        )
        blue_found = sum(
            1 for word in self.revealed
            if self.card_types[word] == CardType.BLUE
        )

        # Total agents per team
        red_total = sum(1 for ct in self.card_types.values() if ct == CardType.RED)
        blue_total = sum(1 for ct in self.card_types.values() if ct == CardType.BLUE)

        return {
            "red": {
                "found": red_found,
                "total": red_total,
                "remaining": self.red_remaining,
            },
            "blue": {
                "found": blue_found,
                "total": blue_total,
                "remaining": self.blue_remaining,
            },
            "winner": self._winner.value if self._winner else None,
            "loss_reason": self._loss_reason,
        }

    def serialize(self) -> Dict[str, Any]:
        """Serialize complete game state."""
        return {
            "game_type": "codenames",
            "seed": self._seed,
            "players": {
                "red_spymaster": self.red_spymaster,
                "red_operatives": self.red_operatives,
                "blue_spymaster": self.blue_spymaster,
                "blue_operatives": self.blue_operatives,
            },
            "grid": self.grid,
            "card_types": {w: ct.value for w, ct in self.card_types.items()},
            "revealed": list(self.revealed),
            "current_team": self.current_team.value,
            "current_phase": self.current_phase.value,
            "current_clue": self.current_clue,
            "guesses_remaining": self.guesses_remaining,
            "turn_number": self.turn_number,
            "clue_history": [
                {"team": c.team.value, "word": c.word, "number": c.number, "turn": c.turn_number}
                for c in self.clue_history
            ],
            "guess_history": [
                {"team": g.team.value, "player": g.player_id, "word": g.word,
                 "card_type": g.card_type.value, "correct": g.correct, "turn": g.turn_number}
                for g in self.guess_history
            ],
            "game_over": self._game_over,
            "winner": self._winner.value if self._winner else None,
            "loss_reason": self._loss_reason,
        }
