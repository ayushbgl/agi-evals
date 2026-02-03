"""Tests for the Avalon game engine.

Coverage:
- Initialisation & determinism
- Phase transitions (proposal -> vote -> mission -> next round)
- Rejection counting, force-approve (4 rejections), 5-rejection spies-win
- Mission success / failure, double-fail mission-4 rule
- All win conditions (3 successes, 3 failures, assassination, 5 rejections)
- Action validation (wrong phase, bad team size, duplicates, non-spy Fail …)
- Private-state / role-knowledge correctness
- Public-log redaction of mission cards
- Serialisation completeness
- Player-ID / index interoperability
"""

from typing import Optional

import pytest

from games.avalon.game import AvalonGame, Phase
from games.avalon.roles import Role, SPY_ROLES, is_spy, mission_team_size, requires_two_fails


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_game(n: int = 5, seed: int = 42, roles=None) -> AvalonGame:
    players = [f"player_{i}" for i in range(n)]
    return AvalonGame(players=players, seed=seed, roles=roles)


def _find_role(game: AvalonGame, role: Role) -> str:
    """Return the player ID assigned *role*."""
    for pid, r in game._role_assignments.items():
        if r == role:
            return pid
    raise ValueError(f"Role {role} not present")


def _vote_all(game: AvalonGame, approve: bool = True) -> None:
    """Drive all votes for the current proposal."""
    while game._phase == Phase.TEAM_VOTE and not game._game_over:
        game.step({"action_type": "VOTE", "approve": approve})


def _play_mission(game: AvalonGame, fail_players: Optional[set] = None) -> None:
    """Drive all mission cards; players in *fail_players* play Fail."""
    fail_players = fail_players or set()
    while game._phase == Phase.MISSION and not game._game_over:
        player = game.get_current_player()
        card = "Fail" if player in fail_players else "Success"
        game.step({"action_type": "PLAY_MISSION_CARD", "card": card})


def _spy_indices(game: AvalonGame):
    return [i for i, pid in enumerate(game._players) if is_spy(game._role_assignments[pid])]


def _resistance_indices(game: AvalonGame):
    return [i for i, pid in enumerate(game._players) if not is_spy(game._role_assignments[pid])]


# ── TestAvalonInit ────────────────────────────────────────────────────────────


class TestAvalonInit:
    def test_5_players(self):
        game = _make_game(5, seed=0)
        assert len(game._role_assignments) == 5
        assert game._phase == Phase.TEAM_PROPOSAL
        assert not game._game_over

    def test_10_players(self):
        game = _make_game(10, seed=0)
        assert len(game._role_assignments) == 10

    def test_too_few_raises(self):
        with pytest.raises(ValueError):
            _make_game(4)

    def test_too_many_raises(self):
        with pytest.raises(ValueError):
            _make_game(11)

    def test_deterministic_same_seed(self):
        g1 = _make_game(5, seed=123)
        g2 = _make_game(5, seed=123)
        assert g1._role_assignments == g2._role_assignments
        assert g1._leader_start_idx == g2._leader_start_idx

    def test_different_seeds_differ(self):
        g1 = _make_game(5, seed=1)
        g2 = _make_game(5, seed=2)
        assert (
            g1._role_assignments != g2._role_assignments
            or g1._leader_start_idx != g2._leader_start_idx
        )

    def test_spy_counts_per_player_count(self):
        expected = {5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 4}
        for n, exp_spies in expected.items():
            game = _make_game(n, seed=0)
            spies = sum(1 for r in game._role_assignments.values() if is_spy(r))
            assert spies == exp_spies

    def test_custom_roles(self):
        roles = ["Merlin", "Percival", "Loyal Servant", "Morgana", "Assassin"]
        game = _make_game(5, seed=0, roles=roles)
        assert set(r.value for r in game._role_assignments.values()) == set(roles)

    def test_custom_roles_wrong_count_raises(self):
        with pytest.raises(ValueError):
            _make_game(5, seed=0, roles=["Merlin", "Morgana", "Assassin"])


# ── TestAvalonPhases ──────────────────────────────────────────────────────────


class TestAvalonPhases:
    def test_starts_in_proposal(self):
        game = _make_game(5, seed=0)
        assert game._phase == Phase.TEAM_PROPOSAL
        assert game.get_current_player() == game._current_leader()

    def test_proposal_to_vote(self):
        game = _make_game(5, seed=0)
        size = game._required_team_size()
        game.step({"action_type": "PROPOSE_TEAM", "team": list(range(size))})
        assert game._phase == Phase.TEAM_VOTE

    def test_approved_vote_to_mission(self):
        game = _make_game(5, seed=0)
        size = game._required_team_size()
        game.step({"action_type": "PROPOSE_TEAM", "team": list(range(size))})
        _vote_all(game, approve=True)
        assert game._phase == Phase.MISSION

    def test_rejected_vote_back_to_proposal(self):
        game = _make_game(5, seed=0)
        size = game._required_team_size()
        game.step({"action_type": "PROPOSE_TEAM", "team": list(range(size))})
        _vote_all(game, approve=False)
        assert game._phase == Phase.TEAM_PROPOSAL
        assert game._consecutive_rejections == 1

    def test_leader_advances_on_rejection(self):
        game = _make_game(5, seed=0)
        leader_before = game._current_leader_idx % game._num_players
        size = game._required_team_size()
        game.step({"action_type": "PROPOSE_TEAM", "team": list(range(size))})
        _vote_all(game, approve=False)
        assert game._current_leader_idx % game._num_players == (leader_before + 1) % 5

    def test_leader_advances_after_mission(self):
        game = _make_game(5, seed=0)
        leader_before = game._current_leader_idx % game._num_players
        size = game._required_team_size()
        game.step({"action_type": "PROPOSE_TEAM", "team": list(range(size))})
        _vote_all(game, approve=True)
        _play_mission(game)
        assert game._current_leader_idx % game._num_players == (leader_before + 1) % 5

    def test_all_players_vote(self):
        game = _make_game(5, seed=0)
        size = game._required_team_size()
        game.step({"action_type": "PROPOSE_TEAM", "team": list(range(size))})
        voted = []
        while game._phase == Phase.TEAM_VOTE:
            voted.append(game.get_current_player())
            game.step({"action_type": "VOTE", "approve": True})
        assert set(voted) == set(game._players)


# ── TestAvalonRejections ──────────────────────────────────────────────────────


class TestAvalonRejections:
    def _reject_n(self, game: AvalonGame, n: int) -> None:
        for _ in range(n):
            if game._game_over:
                break
            size = game._required_team_size()
            game.step({"action_type": "PROPOSE_TEAM", "team": list(range(size))})
            if game._phase == Phase.TEAM_VOTE:
                _vote_all(game, approve=False)

    def test_counter_increments(self):
        game = _make_game(5, seed=0)
        self._reject_n(game, 3)
        assert game._consecutive_rejections == 3

    def test_force_approve_after_four(self):
        game = _make_game(5, seed=0)
        self._reject_n(game, 4)
        assert game._consecutive_rejections == 4
        size = game._required_team_size()
        result, _ = game.step({"action_type": "PROPOSE_TEAM", "team": list(range(size))})
        assert result.get("force_approved") is True
        assert game._phase == Phase.MISSION

    def test_five_rejections_spies_win(self):
        """Directly set up 4 rejections then reject one more via vote."""
        game = _make_game(5, seed=0)
        size = game._required_team_size()
        # Seed the state: pretend 4 rejections already happened, enter vote.
        game._consecutive_rejections = 4
        game._phase = Phase.TEAM_VOTE
        game._voters_remaining = list(game._players)
        game._votes = {}
        game._current_proposal = [game._players[i] for i in range(size)]
        _vote_all(game, approve=False)
        assert game._game_over
        assert game._winner == "spies"
        assert game._win_reason == "five_rejections"

    def test_rejections_reset_after_mission(self):
        game = _make_game(5, seed=0)
        self._reject_n(game, 2)
        assert game._consecutive_rejections == 2
        size = game._required_team_size()
        game.step({"action_type": "PROPOSE_TEAM", "team": list(range(size))})
        _vote_all(game, approve=True)
        _play_mission(game)
        assert game._consecutive_rejections == 0


# ── TestAvalonMissions ────────────────────────────────────────────────────────


class TestAvalonMissions:
    def test_all_success(self):
        game = _make_game(5, seed=0)
        size = game._required_team_size()
        game.step({"action_type": "PROPOSE_TEAM", "team": list(range(size))})
        _vote_all(game, approve=True)
        _play_mission(game)  # all Success
        assert game._resistance_wins == 1
        assert game._spy_wins == 0

    def test_one_fail_fails_mission(self):
        game = _make_game(5, seed=0)
        size = game._required_team_size()
        spies = _spy_indices(game)
        others = _resistance_indices(game)
        team = spies[:1] + others[: size - 1]
        game.step({"action_type": "PROPOSE_TEAM", "team": team})
        _vote_all(game, approve=True)
        _play_mission(game, fail_players={game._players[spies[0]]})
        assert game._spy_wins == 1

    def test_resistance_cannot_play_fail(self):
        game = _make_game(5, seed=0)
        others = _resistance_indices(game)
        size = game._required_team_size()
        team = others[:size]
        game.step({"action_type": "PROPOSE_TEAM", "team": team})
        _vote_all(game, approve=True)
        # First player is resistance (all-resistance team)
        with pytest.raises(ValueError, match="Only spies"):
            game.step({"action_type": "PLAY_MISSION_CARD", "card": "Fail"})

    def test_double_fail_mission4_7players(self):
        """Mission 4 with 7 players: 1 Fail is not enough."""
        game = _make_game(7, seed=0)
        # Fast-forward to mission 4 by directly setting state.
        game._mission_number = 4
        game._resistance_wins = 2
        game._spy_wins = 1
        size = mission_team_size(7, 4)  # == 4
        # Build a team with exactly 1 spy.
        spies = _spy_indices(game)
        others = _resistance_indices(game)
        team_ids = [game._players[spies[0]]] + [game._players[i] for i in others[: size - 1]]
        game._current_proposal = team_ids
        game._phase = Phase.MISSION
        game._mission_players_remaining = [p for p in game._players if p in team_ids]
        game._mission_cards = {}

        # Play: 1 Fail, rest Success.
        for p in game._mission_players_remaining[:]:
            if p == game._players[spies[0]]:
                game.step({"action_type": "PLAY_MISSION_CARD", "card": "Fail"})
            else:
                game.step({"action_type": "PLAY_MISSION_CARD", "card": "Success"})

        # Only 1 Fail when 2 needed -> mission SUCCEEDS
        assert game._mission_results[-1]["mission_failed"] is False
        assert game._resistance_wins == 3  # was 2 + 1 new success

    def test_double_fail_mission4_two_fails_fails(self):
        """Mission 4 with 7 players: 2 Fails does fail the mission."""
        game = _make_game(7, seed=0)
        game._mission_number = 4
        game._resistance_wins = 1
        game._spy_wins = 1
        size = mission_team_size(7, 4)  # == 4

        spies = _spy_indices(game)
        others = _resistance_indices(game)
        # Need 2 spies on team; if only 1 spy exists for this seed, skip
        if len(spies) < 2:
            pytest.skip("Need at least 2 spies for this test")
        team_ids = [game._players[spies[0]], game._players[spies[1]]] + [
            game._players[i] for i in others[: size - 2]
        ]
        game._current_proposal = team_ids
        game._phase = Phase.MISSION
        game._mission_players_remaining = [p for p in game._players if p in team_ids]
        game._mission_cards = {}

        fail_set = {game._players[spies[0]], game._players[spies[1]]}
        for p in game._mission_players_remaining[:]:
            card = "Fail" if p in fail_set else "Success"
            game.step({"action_type": "PLAY_MISSION_CARD", "card": card})

        assert game._mission_results[-1]["mission_failed"] is True
        assert game._spy_wins == 2


# ── TestAvalonWinConditions ───────────────────────────────────────────────────


class TestAvalonWinConditions:
    def _run_successful_mission(self, game: AvalonGame) -> None:
        """Propose an all-resistance team, approve, run with all Success."""
        size = mission_team_size(game._num_players, game._mission_number)
        others = _resistance_indices(game)
        if len(others) < size:
            # Pad with spies but don't fail
            team = others + _spy_indices(game)[: size - len(others)]
        else:
            team = others[:size]
        game.step({"action_type": "PROPOSE_TEAM", "team": team})
        _vote_all(game, approve=True)
        _play_mission(game)  # nobody fails

    def _run_failed_mission(self, game: AvalonGame) -> None:
        """Propose a team with a spy, approve, spy plays Fail."""
        size = mission_team_size(game._num_players, game._mission_number)
        spies = _spy_indices(game)
        others = _resistance_indices(game)
        team = spies[:1] + others[: size - 1]
        spy_pid = game._players[spies[0]]
        game.step({"action_type": "PROPOSE_TEAM", "team": team})
        _vote_all(game, approve=True)
        _play_mission(game, fail_players={spy_pid})

    def test_spies_win_three_failures(self):
        game = _make_game(5, seed=0)
        for _ in range(3):
            if game._game_over:
                break
            self._run_failed_mission(game)
        assert game._game_over
        assert game._winner == "spies"
        assert game._win_reason == "three_failures"

    def test_resistance_wins_no_assassin(self):
        """3 successes without an Assassin -> immediate resistance win."""
        roles = ["Merlin", "Percival", "Loyal Servant", "Morgana", "Minion"]
        game = _make_game(5, seed=0, roles=roles)
        assert Role.ASSASSIN not in game._role_assignments.values()
        for _ in range(3):
            if game._game_over:
                break
            self._run_successful_mission(game)
        assert game._game_over
        assert game._winner == "resistance"
        assert game._win_reason == "three_successes"

    def test_assassination_phase_reached(self):
        """Default 5-player pool has Assassin; 3 successes -> assassination."""
        game = _make_game(5, seed=0)
        assert Role.ASSASSIN in game._role_assignments.values()
        for _ in range(3):
            if game._game_over or game._phase == Phase.ASSASSINATION:
                break
            self._run_successful_mission(game)
        if not game._game_over:
            assert game._phase == Phase.ASSASSINATION

    def test_assassination_success(self):
        game = _make_game(5, seed=0)
        for _ in range(3):
            if game._game_over or game._phase == Phase.ASSASSINATION:
                break
            self._run_successful_mission(game)
        if game._game_over:
            pytest.skip("Game ended before assassination (unexpected role set)")
        assert game._phase == Phase.ASSASSINATION
        merlin_idx = game._players.index(_find_role(game, Role.MERLIN))
        result, _ = game.step({"action_type": "ASSASSINATE", "target": merlin_idx})
        assert result["successful"] is True
        assert game._winner == "spies"
        assert game._win_reason == "assassination_success"

    def test_assassination_failure(self):
        game = _make_game(5, seed=0)
        for _ in range(3):
            if game._game_over or game._phase == Phase.ASSASSINATION:
                break
            self._run_successful_mission(game)
        if game._game_over:
            pytest.skip("Game ended before assassination")
        assert game._phase == Phase.ASSASSINATION
        merlin_pid = _find_role(game, Role.MERLIN)
        assassin_pid = game.get_current_player()
        # Pick any player that is not Merlin and not the Assassin itself
        target_idx = next(
            i for i, pid in enumerate(game._players)
            if pid != merlin_pid and pid != assassin_pid
        )
        result, _ = game.step({"action_type": "ASSASSINATE", "target": target_idx})
        assert result["successful"] is False
        assert game._winner == "resistance"
        assert game._win_reason == "assassination_failed"


# ── TestAvalonValidation ──────────────────────────────────────────────────────


class TestAvalonValidation:
    def test_step_after_game_over(self):
        game = _make_game(5, seed=0)
        game._end_game("spies", "test")
        with pytest.raises(ValueError, match="already over"):
            game.step({"action_type": "VOTE", "approve": True})

    def test_unknown_action_type(self):
        game = _make_game(5, seed=0)
        with pytest.raises(ValueError, match="Unknown action_type"):
            game.step({"action_type": "INVALID"})

    def test_wrong_team_size(self):
        game = _make_game(5, seed=0)
        with pytest.raises(ValueError, match="Team size"):
            game.step({"action_type": "PROPOSE_TEAM", "team": [0]})

    def test_duplicate_players_in_team(self):
        game = _make_game(5, seed=0)
        size = game._required_team_size()
        with pytest.raises(ValueError, match="Duplicate"):
            game.step({"action_type": "PROPOSE_TEAM", "team": [0] * size})

    def test_out_of_range_index(self):
        game = _make_game(5, seed=0)
        size = game._required_team_size()
        with pytest.raises(ValueError, match="out of range"):
            game.step({"action_type": "PROPOSE_TEAM", "team": [99] + list(range(size - 1))})

    def test_vote_wrong_phase(self):
        game = _make_game(5, seed=0)
        with pytest.raises(ValueError, match="team_vote"):
            game.step({"action_type": "VOTE", "approve": True})

    def test_non_bool_approve(self):
        game = _make_game(5, seed=0)
        size = game._required_team_size()
        game.step({"action_type": "PROPOSE_TEAM", "team": list(range(size))})
        with pytest.raises(ValueError, match="boolean"):
            game.step({"action_type": "VOTE", "approve": "yes"})

    def test_assassinate_wrong_phase(self):
        game = _make_game(5, seed=0)
        with pytest.raises(ValueError, match="assassination"):
            game.step({"action_type": "ASSASSINATE", "target": 0})

    def test_assassinate_self(self):
        game = _make_game(5, seed=0)
        game._phase = Phase.ASSASSINATION
        assassin_pid = game.get_current_player()
        assassin_idx = game._players.index(assassin_pid)
        with pytest.raises(ValueError, match="cannot target themselves"):
            game.step({"action_type": "ASSASSINATE", "target": assassin_idx})

    def test_invalid_card_value(self):
        game = _make_game(5, seed=0)
        size = game._required_team_size()
        game.step({"action_type": "PROPOSE_TEAM", "team": list(range(size))})
        _vote_all(game, approve=True)
        with pytest.raises(ValueError, match="'card' must be"):
            game.step({"action_type": "PLAY_MISSION_CARD", "card": "Sabotage"})


# ── TestAvalonStateViews ──────────────────────────────────────────────────────


class TestAvalonStateViews:
    def test_merlin_sees_assassin_not_morgana(self):
        game = _make_game(5, seed=0)
        merlin_pid = _find_role(game, Role.MERLIN)
        priv = game.get_private_state(merlin_pid)
        assert priv["role"] == "Merlin"
        assert priv["alignment"] == "resistance"
        suspects = {pid for pid, lbl in priv["knowledge"].items() if lbl == "suspect"}
        assert _find_role(game, Role.ASSASSIN) in suspects
        assert _find_role(game, Role.MORGANA) not in suspects

    def test_percival_sees_merlin_and_morgana(self):
        game = _make_game(5, seed=0)
        percival_pid = _find_role(game, Role.PERCIVAL)
        priv = game.get_private_state(percival_pid)
        possible = {pid for pid, lbl in priv["knowledge"].items() if lbl == "possible_merlin"}
        assert possible == {_find_role(game, Role.MERLIN), _find_role(game, Role.MORGANA)}

    def test_spy_sees_fellow_spies(self):
        game = _make_game(5, seed=0)
        morgana_pid = _find_role(game, Role.MORGANA)
        priv = game.get_private_state(morgana_pid)
        allies = {pid for pid, lbl in priv["knowledge"].items() if lbl == "fellow_spy"}
        assert _find_role(game, Role.ASSASSIN) in allies

    def test_loyal_servant_empty_knowledge(self):
        game = _make_game(5, seed=0)
        for pid, role in game._role_assignments.items():
            if role == Role.LOYAL_SERVANT:
                assert game.get_private_state(pid)["knowledge"] == {}
                return
        pytest.skip("No Loyal Servant in this game")

    def test_mission_cards_hidden_in_public_log(self):
        game = _make_game(5, seed=0)
        size = game._required_team_size()
        spies = _spy_indices(game)
        others = _resistance_indices(game)
        team = spies[:1] + others[: size - 1]
        game.step({"action_type": "PROPOSE_TEAM", "team": team})
        _vote_all(game, approve=True)
        _play_mission(game, fail_players={game._players[spies[0]]})

        pub = game.get_public_state()
        for entry in pub["action_log"]:
            if entry["action"].get("action_type") == "PLAY_MISSION_CARD":
                assert "card" not in entry["action"]

    def test_full_log_has_cards(self):
        """serialize() action_log retains card values for replay."""
        game = _make_game(5, seed=0)
        size = game._required_team_size()
        spies = _spy_indices(game)
        others = _resistance_indices(game)
        team = spies[:1] + others[: size - 1]
        game.step({"action_type": "PROPOSE_TEAM", "team": team})
        _vote_all(game, approve=True)
        _play_mission(game, fail_players={game._players[spies[0]]})

        full_log = game.serialize()["action_log"]
        mission_actions = [
            e for e in full_log if e["action"].get("action_type") == "PLAY_MISSION_CARD"
        ]
        cards = [e["action"]["card"] for e in mission_actions]
        assert "Fail" in cards

    def test_unknown_player_raises(self):
        game = _make_game(5, seed=0)
        with pytest.raises(ValueError, match="Unknown player"):
            game.get_private_state("ghost")

    def test_get_teams(self):
        game = _make_game(5, seed=0)
        teams = game.get_teams()
        assert len(teams["resistance"]) + len(teams["spies"]) == 5
        for pid in teams["resistance"]:
            assert not is_spy(game._role_assignments[pid])
        for pid in teams["spies"]:
            assert is_spy(game._role_assignments[pid])


# ── TestAvalonAvailableActions ────────────────────────────────────────────────


class TestAvalonAvailableActions:
    def test_proposal_phase(self):
        game = _make_game(5, seed=0)
        actions = game.get_available_actions()
        assert len(actions) == 1
        assert actions[0]["action_type"] == "PROPOSE_TEAM"
        assert actions[0]["team_size"] == game._required_team_size()

    def test_vote_phase(self):
        game = _make_game(5, seed=0)
        size = game._required_team_size()
        game.step({"action_type": "PROPOSE_TEAM", "team": list(range(size))})
        actions = game.get_available_actions()
        assert len(actions) == 2
        assert {(a["action_type"], a["approve"]) for a in actions} == {
            ("VOTE", True), ("VOTE", False)
        }

    def test_resistance_only_success(self):
        game = _make_game(5, seed=0)
        others = _resistance_indices(game)
        size = game._required_team_size()
        team = others[:size]
        game.step({"action_type": "PROPOSE_TEAM", "team": team})
        _vote_all(game, approve=True)
        # All players on this team are resistance
        actions = game.get_available_actions()
        assert len(actions) == 1
        assert actions[0]["card"] == "Success"

    def test_spy_gets_both_cards(self):
        game = _make_game(5, seed=0)
        spies = _spy_indices(game)
        others = _resistance_indices(game)
        size = game._required_team_size()
        # Put spy first in player-list order so they act first in mission
        spy_pid = game._players[spies[0]]
        # Build team: spy + resistance, then reorder via _enter_mission
        team = spies[:1] + others[: size - 1]
        game.step({"action_type": "PROPOSE_TEAM", "team": team})
        _vote_all(game, approve=True)
        # Advance until it's the spy's turn
        while game._phase == Phase.MISSION:
            current = game.get_current_player()
            if current == spy_pid:
                actions = game.get_available_actions()
                assert len(actions) == 2
                assert {a["card"] for a in actions} == {"Success", "Fail"}
                return  # test passed
            game.step({"action_type": "PLAY_MISSION_CARD", "card": "Success"})
        pytest.fail("Spy never got a turn")

    def test_game_over_empty(self):
        game = _make_game(5, seed=0)
        game._end_game("spies", "test")
        assert game.get_available_actions() == []


# ── TestAvalonSerialization ───────────────────────────────────────────────────


class TestAvalonSerialization:
    def test_required_fields(self):
        game = _make_game(5, seed=42)
        data = game.serialize()
        required = {
            "game_type", "seed", "players", "role_assignments",
            "leader_start_idx", "current_leader_idx", "mission_number",
            "phase", "consecutive_rejections", "mission_results",
            "resistance_wins", "spy_wins", "game_over", "winner",
            "win_reason", "setup_log", "action_log",
        }
        assert required.issubset(data.keys())

    def test_game_type_avalon(self):
        assert _make_game(5).serialize()["game_type"] == "avalon"

    def test_setup_log(self):
        game = _make_game(5, seed=42)
        setup = game.serialize()["setup_log"]
        assert setup["num_players"] == 5
        assert len(setup["player_ids"]) == 5
        assert len(setup["roles"]) == 5
        assert setup["seed"] == 42


# ── TestAvalonPlayerResolution ────────────────────────────────────────────────


class TestAvalonPlayerResolution:
    def test_propose_with_string_ids(self):
        game = _make_game(5, seed=0)
        size = game._required_team_size()
        team_ids = game._players[:size]
        game.step({"action_type": "PROPOSE_TEAM", "team": team_ids})
        assert game._phase == Phase.TEAM_VOTE

    def test_assassinate_with_string_id(self):
        game = _make_game(5, seed=0)
        game._phase = Phase.ASSASSINATION
        merlin_pid = _find_role(game, Role.MERLIN)
        result, _ = game.step({"action_type": "ASSASSINATE", "target": merlin_pid})
        assert result["successful"] is True


# ── TestAvalonConstants ───────────────────────────────────────────────────────


class TestAvalonConstants:
    def test_mission_sizes(self):
        expected = {
            5: [2, 3, 2, 3, 3], 6: [2, 4, 3, 3, 4], 7: [2, 4, 3, 4, 5],
            8: [3, 4, 4, 5, 5], 9: [3, 4, 4, 5, 5], 10: [3, 5, 4, 6, 5],
        }
        for n, sizes in expected.items():
            for m, size in enumerate(sizes, 1):
                assert mission_team_size(n, m) == size

    def test_two_fails_only_m4_7plus(self):
        for n in range(5, 11):
            for m in range(1, 6):
                assert requires_two_fails(n, m) == (m == 4 and n >= 7)


# ── TestAvalonActionParser ────────────────────────────────────────────────────


class TestAvalonActionParser:
    def setup_method(self):
        from games.avalon.action_parser import AvalonActionParser
        self.parser = AvalonActionParser(fallback_to_random=False)

    def test_parse_propose_json(self):
        raw = '```json\n{"action_type": "PROPOSE_TEAM", "team": [0, 2, 3]}\n```'
        valid = [{"action_type": "PROPOSE_TEAM", "team_size": 3}]
        result = self.parser.parse(raw, valid)
        assert result == {"action_type": "PROPOSE_TEAM", "team": [0, 2, 3]}

    def test_parse_vote_true(self):
        raw = '{"action_type": "VOTE", "approve": true}'
        valid = [{"action_type": "VOTE", "approve": True}, {"action_type": "VOTE", "approve": False}]
        result = self.parser.parse(raw, valid)
        assert result == {"action_type": "VOTE", "approve": True}

    def test_parse_vote_string_yes(self):
        raw = '{"action_type": "VOTE", "approve": "yes"}'
        valid = [{"action_type": "VOTE", "approve": True}, {"action_type": "VOTE", "approve": False}]
        result = self.parser.parse(raw, valid)
        assert result == {"action_type": "VOTE", "approve": True}

    def test_parse_mission_success(self):
        raw = '{"action_type": "PLAY_MISSION_CARD", "card": "Success"}'
        valid = [{"action_type": "PLAY_MISSION_CARD", "card": "Success"}]
        result = self.parser.parse(raw, valid)
        assert result == {"action_type": "PLAY_MISSION_CARD", "card": "Success"}

    def test_parse_mission_fail_allowed(self):
        raw = '{"action_type": "PLAY_MISSION_CARD", "card": "Fail"}'
        valid = [
            {"action_type": "PLAY_MISSION_CARD", "card": "Success"},
            {"action_type": "PLAY_MISSION_CARD", "card": "Fail"},
        ]
        result = self.parser.parse(raw, valid)
        assert result == {"action_type": "PLAY_MISSION_CARD", "card": "Fail"}

    def test_parse_mission_fail_blocked_for_resistance(self):
        """If valid_actions has no Fail, parser downgrades Fail -> Success."""
        raw = '{"action_type": "PLAY_MISSION_CARD", "card": "Fail"}'
        valid = [{"action_type": "PLAY_MISSION_CARD", "card": "Success"}]
        result = self.parser.parse(raw, valid)
        assert result["card"] == "Success"

    def test_parse_assassinate(self):
        raw = '{"action_type": "ASSASSINATE", "target": 3}'
        valid = [{"action_type": "ASSASSINATE", "valid_targets": [1, 2, 3, 4]}]
        result = self.parser.parse(raw, valid)
        assert result == {"action_type": "ASSASSINATE", "target": 3}

    def test_nl_vote_approve(self):
        raw = "I think this team is good. I approve the team."
        valid = [{"action_type": "VOTE", "approve": True}, {"action_type": "VOTE", "approve": False}]
        result = self.parser.parse(raw, valid)
        assert result == {"action_type": "VOTE", "approve": True}

    def test_nl_vote_reject(self):
        raw = "Something feels off. I reject this proposal."
        valid = [{"action_type": "VOTE", "approve": True}, {"action_type": "VOTE", "approve": False}]
        result = self.parser.parse(raw, valid)
        assert result == {"action_type": "VOTE", "approve": False}

    def test_unparseable_raises(self):
        from core.action_parser import ActionParseError
        raw = "I have no idea what to do"
        valid = [{"action_type": "PROPOSE_TEAM", "team_size": 2}]
        with pytest.raises(ActionParseError):
            self.parser.parse(raw, valid)


# ── TestAvalonStateAdapter ────────────────────────────────────────────────────


class TestAvalonStateAdapter:
    def setup_method(self):
        from games.avalon.state_adapter import AvalonStateAdapter
        self.adapter = AvalonStateAdapter()

    def test_system_prompt_for_merlin(self):
        prompt = self.adapter.format_system_prompt(role="Merlin")
        assert "Merlin" in prompt
        assert "Assassin" in prompt  # assassination warning

    def test_system_prompt_fallback(self):
        prompt = self.adapter.format_system_prompt(role="UnknownRole")
        assert "Avalon" in prompt

    def test_state_to_prompt_contains_role(self):
        game = _make_game(5, seed=0)
        player = game._players[0]
        pub = game.get_public_state()
        priv = game.get_private_state(player)
        valid = game.get_available_actions()
        prompt = self.adapter.state_to_prompt(pub, priv, valid)
        assert priv["role"] in prompt
        assert "Player 0" in prompt

    def test_output_schema_has_four_actions(self):
        schema = self.adapter.get_output_schema()
        assert "oneOf" in schema
        assert len(schema["oneOf"]) == 4
        types = {item["properties"]["action_type"]["enum"][0] for item in schema["oneOf"]}
        assert types == {"PROPOSE_TEAM", "VOTE", "PLAY_MISSION_CARD", "ASSASSINATE"}
