"""
Abstract Game interface for the Game Arena platform.

All games must implement this interface to be compatible with the
LLM evaluation arena orchestration system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple


class Game(ABC):
    """
    Abstract base class for all turn-based games.

    This interface defines the contract that all game implementations must
    follow to integrate with the arena's orchestration system. The design
    supports both simple games (like Codenames) and complex games (like Catan)
    with varying numbers of players, roles, and action types.

    Key concepts:
    - **Players**: Identified by string IDs (e.g., "player_0", "red_spymaster")
    - **Roles**: Game-specific roles (e.g., "player", "spymaster", "operative")
    - **State**: Divided into public (visible to all) and private (per-player)
    - **Actions**: Structured dicts with game-specific fields

    Example implementation:
        class MyGame(Game):
            def reset(self, seed=None):
                self.board = initialize_board(seed)
                self.current_player_idx = 0

            def get_current_player(self) -> str:
                return f"player_{self.current_player_idx}"
    """

    @property
    @abstractmethod
    def game_type(self) -> str:
        """
        Return the game type identifier.

        Returns:
            String identifier for this game type (e.g., "catan", "codenames")
        """
        pass

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset the game to its initial state.

        This should initialize all game state, shuffle decks/tiles if applicable,
        and prepare for the first player's turn.

        Args:
            seed: Optional random seed for reproducible game setup
        """
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Get the complete current game state.

        Returns a comprehensive snapshot of the game including both public
        and all private information. This is useful for serialization and
        replay systems.

        Returns:
            Dict containing:
                - public: Public game state
                - private_states: Dict mapping player_id to their private state
                - metadata: Turn number, phase, etc.
        """
        pass

    @abstractmethod
    def get_public_state(self) -> Dict[str, Any]:
        """
        Get the publicly visible game state.

        This includes information that all players can see, such as:
        - Board/grid layout
        - Revealed cards or pieces
        - Score summaries
        - Turn information

        Returns:
            Dict with game-specific public state fields
        """
        pass

    @abstractmethod
    def get_private_state(self, player_id: str) -> Dict[str, Any]:
        """
        Get private state visible only to a specific player.

        This includes information only this player should know:
        - Hand cards
        - Secret objectives
        - Role-specific information (e.g., spymaster sees card colors)

        Args:
            player_id: The player's unique identifier

        Returns:
            Dict with player-specific private state
        """
        pass

    @abstractmethod
    def get_current_player(self) -> str:
        """
        Get the ID of the player whose turn it currently is.

        Returns:
            Player ID string (e.g., "player_0", "red_spymaster")
        """
        pass

    @abstractmethod
    def get_current_role(self) -> str:
        """
        Get the current player's role in the game.

        Roles help determine which prompts and action types are valid.
        Examples:
        - Catan: Always "player"
        - Codenames: "spymaster" or "operative"
        - Werewolf: "villager", "werewolf", "seer", etc.

        Returns:
            Role identifier string
        """
        pass

    @abstractmethod
    def get_available_actions(self) -> List[Dict[str, Any]]:
        """
        Get all valid actions for the current player.

        Each action is a dict with game-specific fields. The structure should
        match what the step() method expects.

        Returns:
            List of valid action dicts, each containing:
                - action_type: String identifying the action type
                - Additional fields specific to the action type
        """
        pass

    @abstractmethod
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """
        Execute an action and advance the game state.

        Args:
            action: Action dict matching format from get_available_actions()

        Returns:
            Tuple of (result_dict, game_over):
                - result_dict: Outcome of the action (e.g., dice roll, card drawn)
                - game_over: True if the game has ended

        Raises:
            ValueError: If action is invalid or not in available actions
        """
        pass

    @abstractmethod
    def is_over(self) -> bool:
        """
        Check if the game has ended.

        Returns:
            True if game has reached a terminal state (win, loss, or draw)
        """
        pass

    @abstractmethod
    def get_winner(self) -> Optional[str]:
        """
        Get the winner of the game.

        Returns:
            - Player/team ID string if there's a winner
            - None if game is ongoing or ended in a draw
        """
        pass

    @abstractmethod
    def get_scores(self) -> Dict[str, Any]:
        """
        Get current or final scores for all players/teams.

        Returns:
            Dict mapping player/team IDs to their scores.
            Score format is game-specific (int, float, or nested dict).
        """
        pass

    def get_players(self) -> List[str]:
        """
        Get list of all player IDs in the game.

        Default implementation returns an empty list.
        Override to provide actual player list.

        Returns:
            List of player ID strings
        """
        return []

    def get_teams(self) -> Optional[Dict[str, List[str]]]:
        """
        Get team assignments if this is a team-based game.

        Returns:
            Dict mapping team_id to list of player_ids, or None for non-team games
        """
        return None

    def serialize(self) -> Dict[str, Any]:
        """
        Serialize the complete game state for storage/replay.

        Default implementation uses get_state(). Override for
        custom serialization logic.

        Returns:
            JSON-serializable dict representing complete game state
        """
        return self.get_state()

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "Game":
        """
        Restore a game from serialized state.

        Args:
            data: Serialized state from serialize()

        Returns:
            Restored Game instance

        Raises:
            NotImplementedError: If deserialization not supported
        """
        raise NotImplementedError(
            f"{cls.__name__} does not support deserialization"
        )
