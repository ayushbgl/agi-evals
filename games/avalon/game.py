"""Core Avalon game engine.

Phase state-machine
-------------------
TEAM_PROPOSAL  ->  leader proposes a team
                   +-- consecutive_rejections == 4  ->  force-approve, MISSION
                   +-- else                         ->  TEAM_VOTE

TEAM_VOTE      ->  each player casts one vote (sequential)
                   +-- majority approves   ->  MISSION
                   +-- majority rejects    ->  rejections++
                        +-- rejections >= 5  ->  GAME_OVER (spies)
                        +-- else             ->  advance leader, TEAM_PROPOSAL

MISSION        ->  each team member plays one card (sequential, secret)
                   resolve:
                   +-- 3 resistance wins + Assassin & Merlin present  ->  ASSASSINATION
                   +-- 3 resistance wins (no Assassin or no Merlin)   ->  GAME_OVER (resistance)
                   +-- 3 spy wins                                     ->  GAME_OVER (spies)
                   +-- else  ->  advance leader, next mission

ASSASSINATION  ->  Assassin names one player
                   +-- correct (Merlin)   ->  GAME_OVER (spies)
                   +-- incorrect          ->  GAME_OVER (resistance)
"""

import random
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from core.game import Game
from games.avalon.roles import (
    DEFAULT_ROLE_POOL,
    Role,
    SPY_ROLES,
    compute_knowledge,
    is_spy,
    mission_team_size,
    requires_two_fails,
)


class Phase(Enum):
    """Distinct stages within an Avalon game."""

    TEAM_PROPOSAL = "team_proposal"
    TEAM_VOTE = "team_vote"
    MISSION = "mission"
    ASSASSINATION = "assassination"
    GAME_OVER = "game_over"


class AvalonGame(Game):
    """
    State-based, turn-sequential engine for *The Resistance: Avalon*.

    Design invariants:
    - Deterministic given the same seed **and** action sequence.
    - Never decides on behalf of a player; only validates and applies actions.
    - ``get_current_player()`` always returns a single acting player; multi-player
      sub-phases (vote, mission) are flattened into sequential single-player steps.
    - The full state is serialisable and the action log is sufficient to replay
      the game from scratch.
    """

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        players: List[str],
        seed: Optional[int] = None,
        roles: Optional[List[str]] = None,
    ):
        """
        Args:
            players: Ordered list of unique player IDs (5–10).
            seed:    Seed for reproducible role assignment and leader selection.
                     ``None`` uses an unseeded RNG.
            roles:   Optional explicit role pool (list of role-name strings).
                     ``None`` -> default table for the player count.
        """
        if not (5 <= len(players) <= 10):
            raise ValueError(f"Avalon requires 5\u201310 players, got {len(players)}")
        self._players = list(players)
        self._num_players = len(players)
        self._seed = seed
        self._configured_roles: Optional[List[Role]] = (
            [Role(r) for r in roles] if roles else None
        )
        self.reset(seed)

    # ------------------------------------------------------------------
    # identity
    # ------------------------------------------------------------------

    @property
    def game_type(self) -> str:
        return "avalon"

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> None:
        """Re-initialise the game.  Pass a new seed to change the setup."""
        if seed is not None:
            self._seed = seed
        self._rng = random.Random(self._seed)

        # ── roles ─────────────────────────────────────────────────────
        pool = (
            list(self._configured_roles)
            if self._configured_roles
            else list(DEFAULT_ROLE_POOL[self._num_players])
        )
        if len(pool) != self._num_players:
            raise ValueError(
                f"Role pool length {len(pool)} != player count {self._num_players}"
            )
        self._rng.shuffle(pool)
        self._role_assignments: Dict[str, Role] = dict(zip(self._players, pool))

        # ── leader rotation ───────────────────────────────────────────
        self._leader_start_idx: int = self._rng.randrange(self._num_players)
        self._current_leader_idx: int = self._leader_start_idx

        # ── mission counters ──────────────────────────────────────────
        self._mission_number: int = 1  # 1-based
        self._mission_results: List[Dict[str, Any]] = []
        self._resistance_wins: int = 0
        self._spy_wins: int = 0

        # ── round state ───────────────────────────────────────────────
        self._consecutive_rejections: int = 0
        self._phase: Phase = Phase.TEAM_PROPOSAL
        self._current_proposal: Optional[List[str]] = None

        # ── vote sub-phase ────────────────────────────────────────────
        self._voters_remaining: List[str] = []
        self._votes: Dict[str, bool] = {}

        # ── mission sub-phase ─────────────────────────────────────────
        self._mission_players_remaining: List[str] = []
        self._mission_cards: Dict[str, str] = {}  # player_id -> "Success"|"Fail"

        # ── assassination sub-phase ───────────────────────────────────
        self._assassination_target: Optional[str] = None

        # ── terminal ──────────────────────────────────────────────────
        self._game_over: bool = False
        self._winner: Optional[str] = None   # "resistance" | "spies"
        self._win_reason: Optional[str] = None

        # ── structured log ────────────────────────────────────────────
        self._action_log: List[Dict[str, Any]] = []
        self._setup_log: Dict[str, Any] = {
            "num_players": self._num_players,
            "player_ids": list(self._players),
            "roles": {pid: r.value for pid, r in self._role_assignments.items()},
            "starting_leader": self._players[self._leader_start_idx],
            "seed": self._seed,
        }

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _player_idx(self, player_id: str) -> int:
        return self._players.index(player_id)

    def _current_leader(self) -> str:
        return self._players[self._current_leader_idx % self._num_players]

    def _required_team_size(self) -> int:
        return mission_team_size(self._num_players, self._mission_number)

    def _advance_leader(self) -> None:
        self._current_leader_idx = (self._current_leader_idx + 1) % self._num_players

    def _end_game(self, winner: str, reason: str) -> None:
        self._game_over = True
        self._winner = winner
        self._win_reason = reason
        self._phase = Phase.GAME_OVER

    def _log(self, player_id: str, action: Dict[str, Any], result: Dict[str, Any]) -> None:
        self._action_log.append({
            "step": len(self._action_log),
            "mission": self._mission_number,
            "phase": self._phase.value,
            "player": player_id,
            "action": dict(action),
            "result": dict(result),
        })

    def _resolve_team(self, raw_team: list) -> List[str]:
        """Normalise a list of int indices or str IDs into player IDs."""
        team: List[str] = []
        for member in raw_team:
            if isinstance(member, int):
                if not (0 <= member < self._num_players):
                    raise ValueError(f"Player index out of range: {member}")
                team.append(self._players[member])
            elif isinstance(member, str):
                if member not in self._players:
                    raise ValueError(f"Unknown player ID: {member}")
                team.append(member)
            else:
                raise ValueError(
                    f"Team member must be int (index) or str (ID), got {type(member)}"
                )
        return team

    def _resolve_player(self, raw: Any) -> str:
        """Resolve a single player reference (index or ID) to a player ID."""
        if isinstance(raw, int):
            if not (0 <= raw < self._num_players):
                raise ValueError(f"Player index out of range: {raw}")
            return self._players[raw]
        if isinstance(raw, str):
            if raw not in self._players:
                raise ValueError(f"Unknown player ID: {raw}")
            return raw
        raise ValueError(f"Player reference must be int or str, got {type(raw)}")

    # ------------------------------------------------------------------
    # core Game interface – navigation
    # ------------------------------------------------------------------

    def get_players(self) -> List[str]:
        return list(self._players)

    def get_teams(self) -> Dict[str, List[str]]:
        return {
            "resistance": [pid for pid, r in self._role_assignments.items() if not is_spy(r)],
            "spies":      [pid for pid, r in self._role_assignments.items() if is_spy(r)],
        }

    def get_current_player(self) -> str:
        if self._game_over:
            return ""
        if self._phase == Phase.TEAM_PROPOSAL:
            return self._current_leader()
        if self._phase == Phase.TEAM_VOTE:
            return self._voters_remaining[0] if self._voters_remaining else ""
        if self._phase == Phase.MISSION:
            return self._mission_players_remaining[0] if self._mission_players_remaining else ""
        if self._phase == Phase.ASSASSINATION:
            # Assassin is the acting player; fall back to first spy if missing.
            for pid, role in self._role_assignments.items():
                if role == Role.ASSASSIN:
                    return pid
            return next(pid for pid, r in self._role_assignments.items() if is_spy(r))
        return ""

    def get_current_role(self) -> str:
        player = self.get_current_player()
        if not player:
            return ""
        return self._role_assignments[player].value

    # ------------------------------------------------------------------
    # state views
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        return {
            "public": self.get_public_state(),
            "private_states": {pid: self.get_private_state(pid) for pid in self._players},
            "metadata": {
                "phase": self._phase.value,
                "mission_number": self._mission_number,
                "game_over": self._game_over,
                "winner": self._winner,
            },
        }

    def _public_action_log(self) -> List[Dict[str, Any]]:
        """Return the action log with mission-card values redacted."""
        out = []
        for entry in self._action_log:
            if entry.get("action", {}).get("action_type") == "PLAY_MISSION_CARD":
                sanitised = {k: v for k, v in entry["action"].items() if k != "card"}
                entry = {**entry, "action": sanitised}
            out.append(entry)
        return out

    def get_public_state(self) -> Dict[str, Any]:
        return {
            "players": list(self._players),
            "num_players": self._num_players,
            "phase": self._phase.value,
            "mission_number": self._mission_number,
            "current_leader": self._current_leader() if not self._game_over else None,
            "current_leader_idx": (
                self._current_leader_idx % self._num_players if not self._game_over else None
            ),
            "consecutive_rejections": self._consecutive_rejections,
            "current_proposal": list(self._current_proposal) if self._current_proposal else None,
            "current_proposal_indices": (
                [self._player_idx(p) for p in self._current_proposal]
                if self._current_proposal else None
            ),
            # Votes are public; populated during / after the vote sub-phase.
            "votes": dict(self._votes) if self._votes else None,
            # Completed-mission summaries (no per-player card info).
            "mission_results": list(self._mission_results),
            "resistance_wins": self._resistance_wins,
            "spy_wins": self._spy_wins,
            # Reference tables for the current player count.
            "mission_sizes": [mission_team_size(self._num_players, m) for m in range(1, 6)],
            "requires_two_fails_mission4": requires_two_fails(self._num_players, 4),
            # Terminal.
            "game_over": self._game_over,
            "winner": self._winner,
            "win_reason": self._win_reason,
            # Sanitised action history (cards redacted).
            "action_log": self._public_action_log(),
        }

    def get_private_state(self, player_id: str) -> Dict[str, Any]:
        if player_id not in self._players:
            raise ValueError(f"Unknown player: {player_id}")

        role = self._role_assignments[player_id]
        knowledge = compute_knowledge(player_id, role, self._role_assignments)

        state: Dict[str, Any] = {
            "player_id": player_id,
            "player_index": self._player_idx(player_id),
            "role": role.value,
            "alignment": "spy" if is_spy(role) else "resistance",
            "knowledge": knowledge,
        }

        # Contextual extras.
        if self._phase == Phase.ASSASSINATION and role == Role.ASSASSIN:
            state["spy_allies"] = [pid for pid, lbl in knowledge.items() if lbl == "fellow_spy"]
        if player_id in self._votes:
            state["my_vote"] = self._votes[player_id]
        if player_id in self._mission_cards:
            state["my_mission_card"] = self._mission_cards[player_id]

        return state

    # ------------------------------------------------------------------
    # available actions
    # ------------------------------------------------------------------

    def get_available_actions(self) -> List[Dict[str, Any]]:
        if self._game_over:
            return []

        if self._phase == Phase.TEAM_PROPOSAL:
            size = self._required_team_size()
            return [{
                "action_type": "PROPOSE_TEAM",
                "description": (
                    f"Choose exactly {size} players (indices 0\u2013{self._num_players - 1}) "
                    f"for mission {self._mission_number}."
                ),
                "team_size": size,
            }]

        if self._phase == Phase.TEAM_VOTE:
            return [
                {"action_type": "VOTE", "approve": True},
                {"action_type": "VOTE", "approve": False},
            ]

        if self._phase == Phase.MISSION:
            current = self.get_current_player()
            if is_spy(self._role_assignments[current]):
                return [
                    {"action_type": "PLAY_MISSION_CARD", "card": "Success"},
                    {"action_type": "PLAY_MISSION_CARD", "card": "Fail"},
                ]
            return [{"action_type": "PLAY_MISSION_CARD", "card": "Success"}]

        if self._phase == Phase.ASSASSINATION:
            assassin = self.get_current_player()
            return [{
                "action_type": "ASSASSINATE",
                "description": "Identify which player is Merlin.",
                "valid_targets": [i for i, pid in enumerate(self._players) if pid != assassin],
            }]

        return []

    # ------------------------------------------------------------------
    # step – dispatch
    # ------------------------------------------------------------------

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """
        Validate and apply *action*.

        Returns:
            ``(result_dict, game_over)``

        Raises:
            ValueError: on any constraint violation.
        """
        if self._game_over:
            raise ValueError("Game is already over")

        action_type = action.get("action_type")
        if action_type == "PROPOSE_TEAM":
            return self._handle_propose(action)
        if action_type == "VOTE":
            return self._handle_vote(action)
        if action_type == "PLAY_MISSION_CARD":
            return self._handle_mission_card(action)
        if action_type == "ASSASSINATE":
            return self._handle_assassinate(action)
        raise ValueError(f"Unknown action_type: '{action_type}'")

    # ------------------------------------------------------------------
    # action handlers
    # ------------------------------------------------------------------

    def _handle_propose(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        if self._phase != Phase.TEAM_PROPOSAL:
            raise ValueError("PROPOSE_TEAM is only valid in the team_proposal phase")

        leader = self._current_leader()
        team = self._resolve_team(action.get("team", []))

        required = self._required_team_size()
        if len(team) != required:
            raise ValueError(f"Team size must be exactly {required}, got {len(team)}")
        if len(set(team)) != len(team):
            raise ValueError("Duplicate players in team proposal")

        self._current_proposal = team
        result: Dict[str, Any] = {
            "leader": leader,
            "proposed_team": team,
            "proposed_team_indices": [self._player_idx(p) for p in team],
        }

        if self._consecutive_rejections >= 4:
            # Force-approve: skip the vote entirely.
            result["force_approved"] = True
            self._log(leader, action, result)
            self._enter_mission()
        else:
            self._log(leader, action, result)
            self._phase = Phase.TEAM_VOTE
            self._voters_remaining = list(self._players)
            self._votes = {}

        return result, self._game_over

    def _handle_vote(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        if self._phase != Phase.TEAM_VOTE:
            raise ValueError("VOTE is only valid in the team_vote phase")
        if not self._voters_remaining:
            raise ValueError("All players have already voted")

        voter = self._voters_remaining[0]
        approve = action.get("approve")
        if not isinstance(approve, bool):
            raise ValueError("'approve' must be a boolean (True / False)")

        self._votes[voter] = approve
        self._voters_remaining.pop(0)

        result: Dict[str, Any] = {"voter": voter, "approved": approve}
        self._log(voter, action, result)

        # Once every player has voted, resolve.
        if not self._voters_remaining:
            self._resolve_vote(result)

        return result, self._game_over

    def _resolve_vote(self, result: Dict[str, Any]) -> None:
        approve_count = sum(1 for v in self._votes.values() if v)
        passed = approve_count > self._num_players / 2  # strict majority

        result["vote_resolved"] = True
        result["approve_count"] = approve_count
        result["reject_count"] = self._num_players - approve_count
        result["passed"] = passed

        if passed:
            self._consecutive_rejections = 0
            self._enter_mission()
        else:
            self._consecutive_rejections += 1
            if self._consecutive_rejections >= 5:
                self._end_game("spies", "five_rejections")
            else:
                self._advance_leader()
                self._phase = Phase.TEAM_PROPOSAL
                self._current_proposal = None
                self._votes = {}

    def _handle_mission_card(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        if self._phase != Phase.MISSION:
            raise ValueError("PLAY_MISSION_CARD is only valid in the mission phase")
        if not self._mission_players_remaining:
            raise ValueError("All mission team members have already played")

        player = self._mission_players_remaining[0]
        card = action.get("card")
        if card not in ("Success", "Fail"):
            raise ValueError("'card' must be 'Success' or 'Fail'")
        if card == "Fail" and not is_spy(self._role_assignments[player]):
            raise ValueError("Only spies may play a Fail card")

        self._mission_cards[player] = card
        self._mission_players_remaining.pop(0)

        # Card value is intentionally omitted from the public-facing result.
        result: Dict[str, Any] = {"player": player}
        self._log(player, action, result)

        if not self._mission_players_remaining:
            self._resolve_mission(result)

        return result, self._game_over

    def _resolve_mission(self, result: Dict[str, Any]) -> None:
        fail_count = sum(1 for c in self._mission_cards.values() if c == "Fail")
        two_fails = requires_two_fails(self._num_players, self._mission_number)
        failed = fail_count >= (2 if two_fails else 1)

        summary: Dict[str, Any] = {
            "mission_number": self._mission_number,
            "team": list(self._current_proposal or []),
            "team_indices": [self._player_idx(p) for p in (self._current_proposal or [])],
            "fail_count": fail_count,
            "two_fails_required": two_fails,
            "mission_failed": failed,
        }
        self._mission_results.append(summary)

        if failed:
            self._spy_wins += 1
        else:
            self._resistance_wins += 1

        result["mission_resolved"] = True
        result.update(summary)

        # Check terminal conditions.
        if self._resistance_wins == 3:
            has_assassin = Role.ASSASSIN in self._role_assignments.values()
            has_merlin = Role.MERLIN in self._role_assignments.values()
            if has_assassin and has_merlin:
                self._phase = Phase.ASSASSINATION
            else:
                self._end_game("resistance", "three_successes")
        elif self._spy_wins == 3:
            self._end_game("spies", "three_failures")
        else:
            # Advance to next mission.
            self._mission_number += 1
            self._advance_leader()
            self._consecutive_rejections = 0
            self._phase = Phase.TEAM_PROPOSAL
            self._current_proposal = None
            self._votes = {}
            self._mission_cards = {}

    def _handle_assassinate(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        if self._phase != Phase.ASSASSINATION:
            raise ValueError("ASSASSINATE is only valid in the assassination phase")

        assassin = self.get_current_player()
        target_id = self._resolve_player(action.get("target"))

        if target_id == assassin:
            raise ValueError("The Assassin cannot target themselves")

        self._assassination_target = target_id
        actual_role = self._role_assignments[target_id]
        successful = actual_role == Role.MERLIN

        result: Dict[str, Any] = {
            "assassin": assassin,
            "target": target_id,
            "target_index": self._player_idx(target_id),
            "target_role": actual_role.value,
            "successful": successful,
        }
        self._log(assassin, action, result)

        self._end_game(
            "spies" if successful else "resistance",
            "assassination_success" if successful else "assassination_failed",
        )
        return result, self._game_over

    # ------------------------------------------------------------------
    # internal transitions
    # ------------------------------------------------------------------

    def _enter_mission(self) -> None:
        """Transition into the mission sub-phase."""
        self._phase = Phase.MISSION
        # Preserve original player-list ordering within the mission team.
        self._mission_players_remaining = [
            p for p in self._players if p in (self._current_proposal or [])
        ]
        self._mission_cards = {}

    # ------------------------------------------------------------------
    # terminal queries
    # ------------------------------------------------------------------

    def is_over(self) -> bool:
        return self._game_over

    def get_winner(self) -> Optional[str]:
        return self._winner

    def get_scores(self) -> Dict[str, Any]:
        return {
            "resistance": self._resistance_wins,
            "spies":      self._spy_wins,
            "winner":     self._winner,
            "win_reason": self._win_reason,
        }

    # ------------------------------------------------------------------
    # serialisation
    # ------------------------------------------------------------------

    def serialize(self) -> Dict[str, Any]:
        """Full-fidelity snapshot including secret state for replay / analysis."""
        return {
            "game_type": "avalon",
            "seed": self._seed,
            "players": list(self._players),
            "role_assignments": {pid: r.value for pid, r in self._role_assignments.items()},
            "leader_start_idx": self._leader_start_idx,
            "current_leader_idx": self._current_leader_idx % self._num_players,
            "mission_number": self._mission_number,
            "phase": self._phase.value,
            "consecutive_rejections": self._consecutive_rejections,
            "current_proposal": self._current_proposal,
            "votes": self._votes if self._votes else None,
            "mission_cards": self._mission_cards if self._mission_cards else None,
            "mission_results": self._mission_results,
            "resistance_wins": self._resistance_wins,
            "spy_wins": self._spy_wins,
            "assassination_target": self._assassination_target,
            "game_over": self._game_over,
            "winner": self._winner,
            "win_reason": self._win_reason,
            "setup_log": self._setup_log,
            "action_log": self._action_log,  # full log – cards included
        }
