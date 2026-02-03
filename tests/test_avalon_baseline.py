"""Tests for AvalonBaselineAgent heuristics.

Unit tests use mock state dicts directly — no real game engine is
involved.  The integration tests at the bottom drive a full game
with real AvalonGame instances to verify end-to-end correctness.
"""

import pytest

from games.avalon.baseline_agents import AvalonBaselineAgent
from games.avalon.game import AvalonGame


# ── helpers ────────────────────────────────────────────────────────────────


def _pub(
    *,
    players=None,
    mission_results=None,
    proposal_indices=None,
    **extra,
):
    """Minimal public_state with sensible 5-player defaults."""
    players = players or ["p0", "p1", "p2", "p3", "p4"]
    return {
        "players": players,
        "num_players": len(players),
        "mission_results": mission_results or [],
        "current_proposal_indices": proposal_indices,
        **extra,
    }


def _priv(
    *,
    role="Loyal Servant",
    alignment="resistance",
    player_index=0,
    player_id="p0",
    knowledge=None,
):
    """Minimal private_state with sensible defaults."""
    return {
        "player_id": player_id,
        "player_index": player_index,
        "role": role,
        "alignment": alignment,
        "knowledge": knowledge or {},
    }


def _decide(agent, pub, priv, valid_actions):
    """Call decide() with the standard kwargs wiring."""
    return agent.decide(
        prompt="",
        valid_actions=valid_actions,
        public_state=pub,
        private_state=priv,
    )


# ── constants ──────────────────────────────────────────────────────────────

PROPOSE_2 = [{"action_type": "PROPOSE_TEAM", "team_size": 2}]
PROPOSE_3 = [{"action_type": "PROPOSE_TEAM", "team_size": 3}]
VOTE_ACTIONS = [
    {"action_type": "VOTE", "approve": True},
    {"action_type": "VOTE", "approve": False},
]


# ── PROPOSE_TEAM ───────────────────────────────────────────────────────────


class TestBaselinePropose:
    def test_resistance_avoids_tainted(self):
        """Tainted={0,1}, team_size=3 -> only {2,3,4} remain."""
        pub = _pub(mission_results=[
            {"mission_failed": True, "team_indices": [0, 1]},
        ])
        result = _decide(AvalonBaselineAgent("p2", seed=0), pub, _priv(player_index=2), PROPOSE_3)
        assert set(result["action"]["team"]) == {2, 3, 4}

    def test_resistance_no_history_picks_valid_team(self):
        """No failed missions -> any 2-player team is fine."""
        result = _decide(AvalonBaselineAgent("p0", seed=7), _pub(), _priv(), PROPOSE_2)
        team = result["action"]["team"]
        assert len(team) == 2
        assert len(set(team)) == 2
        assert all(0 <= i < 5 for i in team)

    def test_resistance_soft_constraint_all_tainted(self):
        """All players tainted -> team still formed (avoid is soft)."""
        pub = _pub(mission_results=[
            {"mission_failed": True, "team_indices": [0, 1, 2, 3, 4]},
        ])
        result = _decide(AvalonBaselineAgent("p0", seed=0), pub, _priv(), PROPOSE_2)
        assert len(result["action"]["team"]) == 2

    def test_merlin_avoids_suspects(self):
        """Suspects={3,4}, no tainted -> team drawn from {0,1,2}."""
        priv = _priv(role="Merlin", knowledge={"p3": "suspect", "p4": "suspect"})
        result = _decide(AvalonBaselineAgent("p0", seed=0), _pub(), priv, PROPOSE_2)
        assert set(result["action"]["team"]) <= {0, 1, 2}

    def test_merlin_combines_tainted_and_suspects(self):
        """Tainted={1}, suspect={3} -> avoid={1,3}, team from {0,2,4}."""
        pub = _pub(mission_results=[
            {"mission_failed": True, "team_indices": [1]},
        ])
        priv = _priv(role="Merlin", knowledge={"p3": "suspect"})
        result = _decide(AvalonBaselineAgent("p0", seed=0), pub, priv, PROPOSE_2)
        assert set(result["action"]["team"]) <= {0, 2, 4}

    def test_spy_includes_self(self):
        """Spy idx=0, fellows={1,4}. team_size=2 -> {0, one of 2|3}."""
        priv = _priv(
            role="Morgana", alignment="spy", player_index=0, player_id="p0",
            knowledge={"p1": "fellow_spy", "p4": "fellow_spy"},
        )
        result = _decide(AvalonBaselineAgent("p0", seed=0), _pub(), priv, PROPOSE_2)
        team = result["action"]["team"]
        assert 0 in team
        assert (set(team) - {0}) <= {2, 3}

    def test_spy_includes_self_larger_team(self):
        """Spy idx=3, fellow={1}. team_size=3 -> {3} + 2 from {0,2,4}."""
        priv = _priv(
            role="Assassin", alignment="spy", player_index=3, player_id="p3",
            knowledge={"p1": "fellow_spy"},
        )
        result = _decide(AvalonBaselineAgent("p3", seed=42), _pub(), priv, PROPOSE_3)
        team = result["action"]["team"]
        assert 3 in team
        assert (set(team) - {3}) <= {0, 2, 4}

    def test_percival_ignores_possible_merlin(self):
        """Percival treats possible_merlin same as generic Resistance."""
        pub = _pub(mission_results=[
            {"mission_failed": True, "team_indices": [2]},
        ])
        priv = _priv(
            role="Percival",
            knowledge={"p1": "possible_merlin", "p3": "possible_merlin"},
        )
        result = _decide(AvalonBaselineAgent("p0", seed=0), pub, priv, PROPOSE_2)
        # Only tainted idx 2 is avoided; possible_merlin has no effect.
        assert 2 not in result["action"]["team"]


# ── VOTE ───────────────────────────────────────────────────────────────────


class TestBaselineVote:
    def test_resistance_approves_clean_team(self):
        pub = _pub(
            mission_results=[{"mission_failed": True, "team_indices": [0]}],
            proposal_indices=[1, 2],  # no tainted
        )
        result = _decide(AvalonBaselineAgent("p3", seed=0), pub, _priv(player_index=3), VOTE_ACTIONS)
        assert result["action"]["approve"] is True

    def test_resistance_rejects_tainted_team(self):
        pub = _pub(
            mission_results=[{"mission_failed": True, "team_indices": [0, 1]}],
            proposal_indices=[0, 2],  # 0 is tainted
        )
        result = _decide(AvalonBaselineAgent("p3", seed=0), pub, _priv(player_index=3), VOTE_ACTIONS)
        assert result["action"]["approve"] is False

    def test_resistance_approves_no_history(self):
        """No missions yet -> nothing tainted -> approve."""
        pub = _pub(proposal_indices=[0, 1])
        result = _decide(AvalonBaselineAgent("p2", seed=0), pub, _priv(player_index=2), VOTE_ACTIONS)
        assert result["action"]["approve"] is True

    def test_merlin_rejects_suspect_on_team(self):
        pub = _pub(proposal_indices=[1, 3])
        priv = _priv(role="Merlin", knowledge={"p3": "suspect"})
        result = _decide(AvalonBaselineAgent("p0", seed=0), pub, priv, VOTE_ACTIONS)
        assert result["action"]["approve"] is False

    def test_merlin_falls_through_to_tainted_check(self):
        """Suspect idx=3 not on team; tainted idx=1 IS -> reject."""
        pub = _pub(
            mission_results=[{"mission_failed": True, "team_indices": [1]}],
            proposal_indices=[1, 2],
        )
        priv = _priv(role="Merlin", knowledge={"p3": "suspect"})
        result = _decide(AvalonBaselineAgent("p0", seed=0), pub, priv, VOTE_ACTIONS)
        assert result["action"]["approve"] is False

    def test_merlin_approves_clean_team(self):
        """No suspect and no tainted on team -> approve."""
        pub = _pub(
            mission_results=[{"mission_failed": True, "team_indices": [1]}],
            proposal_indices=[0, 2],  # neither suspect nor tainted
        )
        priv = _priv(role="Merlin", knowledge={"p3": "suspect"})
        result = _decide(AvalonBaselineAgent("p0", seed=0), pub, priv, VOTE_ACTIONS)
        assert result["action"]["approve"] is True

    def test_spy_approves_self_on_team(self):
        pub = _pub(proposal_indices=[2, 3])
        priv = _priv(
            role="Assassin", alignment="spy", player_index=2, player_id="p2",
            knowledge={"p4": "fellow_spy"},
        )
        result = _decide(AvalonBaselineAgent("p2", seed=0), pub, priv, VOTE_ACTIONS)
        assert result["action"]["approve"] is True

    def test_spy_approves_fellow_on_team(self):
        """Self (idx 2) not on team; fellow spy idx=4 IS."""
        pub = _pub(proposal_indices=[0, 4])
        priv = _priv(
            role="Morgana", alignment="spy", player_index=2, player_id="p2",
            knowledge={"p4": "fellow_spy"},
        )
        result = _decide(AvalonBaselineAgent("p2", seed=0), pub, priv, VOTE_ACTIONS)
        assert result["action"]["approve"] is True

    def test_spy_rejects_no_spy_on_team(self):
        """Team=[0,1]; spy_indices={3,4} -> no overlap -> reject."""
        pub = _pub(proposal_indices=[0, 1])
        priv = _priv(
            role="Minion", alignment="spy", player_index=3, player_id="p3",
            knowledge={"p4": "fellow_spy"},
        )
        result = _decide(AvalonBaselineAgent("p3", seed=0), pub, priv, VOTE_ACTIONS)
        assert result["action"]["approve"] is False

    def test_percival_ignores_possible_merlin_in_vote(self):
        """Percival does not reject on possible_merlin labels."""
        pub = _pub(proposal_indices=[1, 3])  # no tainted players
        priv = _priv(
            role="Percival",
            knowledge={"p1": "possible_merlin", "p3": "possible_merlin"},
        )
        result = _decide(AvalonBaselineAgent("p0", seed=0), pub, priv, VOTE_ACTIONS)
        assert result["action"]["approve"] is True


# ── PLAY_MISSION_CARD ──────────────────────────────────────────────────────


class TestBaselineMissionCard:
    RESISTANCE_CARDS = [{"action_type": "PLAY_MISSION_CARD", "card": "Success"}]
    SPY_CARDS = [
        {"action_type": "PLAY_MISSION_CARD", "card": "Success"},
        {"action_type": "PLAY_MISSION_CARD", "card": "Fail"},
    ]

    def test_resistance_plays_success(self):
        result = _decide(
            AvalonBaselineAgent("p0", seed=0),
            _pub(), _priv(alignment="resistance"), self.RESISTANCE_CARDS,
        )
        assert result["action"]["card"] == "Success"

    def test_spy_plays_fail(self):
        result = _decide(
            AvalonBaselineAgent("p0", seed=0),
            _pub(), _priv(role="Morgana", alignment="spy"), self.SPY_CARDS,
        )
        assert result["action"]["card"] == "Fail"


# ── ASSASSINATE ────────────────────────────────────────────────────────────


class TestBaselineAssassinate:
    def test_filters_known_spies(self):
        """Fellows={1,4} filtered out -> target in {0,2}."""
        actions = [{"action_type": "ASSASSINATE", "valid_targets": [0, 1, 2, 4]}]
        priv = _priv(
            role="Assassin", alignment="spy", player_index=3, player_id="p3",
            knowledge={"p1": "fellow_spy", "p4": "fellow_spy"},
        )
        result = _decide(AvalonBaselineAgent("p3", seed=0), _pub(), priv, actions)
        assert result["action"]["target"] in {0, 2}

    def test_fallback_when_all_targets_are_spies(self):
        """Every valid target is a known spy -> still picks one."""
        actions = [{"action_type": "ASSASSINATE", "valid_targets": [1, 4]}]
        priv = _priv(
            role="Assassin", alignment="spy", player_index=3, player_id="p3",
            knowledge={"p1": "fellow_spy", "p4": "fellow_spy"},
        )
        result = _decide(AvalonBaselineAgent("p3", seed=0), _pub(), priv, actions)
        assert result["action"]["target"] in {1, 4}

    def test_no_knowledge_picks_any_target(self):
        """Empty knowledge -> all valid targets are candidates."""
        actions = [{"action_type": "ASSASSINATE", "valid_targets": [0, 1, 2]}]
        priv = _priv(role="Assassin", alignment="spy", player_index=3, player_id="p3")
        result = _decide(AvalonBaselineAgent("p3", seed=42), _pub(), priv, actions)
        assert result["action"]["target"] in {0, 1, 2}


# ── return-value contract ──────────────────────────────────────────────────


class TestBaselineReturnFormat:
    def test_decide_has_required_keys(self):
        result = _decide(
            AvalonBaselineAgent("p0", seed=0),
            _pub(proposal_indices=[0, 1]), _priv(), VOTE_ACTIONS,
        )
        assert {"action", "reasoning", "raw_output", "metadata"} <= result.keys()

    def test_metadata_contains_role_and_method(self):
        result = _decide(
            AvalonBaselineAgent("p0", seed=0),
            _pub(proposal_indices=[0, 1]), _priv(role="Merlin"), VOTE_ACTIONS,
        )
        assert result["metadata"]["role"] == "Merlin"
        assert result["metadata"]["method"] == "rule_based"


# ── integration: full games with baseline agents ──────────────────────────


class TestBaselineIntegration:
    @staticmethod
    def _play(num_players: int, game_seed: int, agent_seed: int = 0):
        """Run a complete game; return the finished AvalonGame."""
        players = [f"p{i}" for i in range(num_players)]
        game = AvalonGame(players=players, seed=game_seed)
        agents = {
            pid: AvalonBaselineAgent(pid, seed=agent_seed + i)
            for i, pid in enumerate(players)
        }
        for _ in range(1000):  # hard cap; real games finish well before this
            if game.is_over():
                break
            pid = game.get_current_player()
            decision = agents[pid].decide(
                prompt="",
                valid_actions=game.get_available_actions(),
                public_state=game.get_public_state(),
                private_state=game.get_private_state(pid),
            )
            game.step(decision["action"])
        return game

    def test_5_player_game_completes(self):
        game = self._play(5, game_seed=99)
        assert game.is_over()
        assert game.get_winner() in ("resistance", "spies")

    def test_10_player_game_completes(self):
        game = self._play(10, game_seed=7)
        assert game.is_over()
        assert game.get_winner() in ("resistance", "spies")

    def test_deterministic_replay(self):
        """Identical seeds produce identical action logs and winner."""
        g1 = self._play(5, game_seed=123, agent_seed=10)
        g2 = self._play(5, game_seed=123, agent_seed=10)
        assert g1.get_winner() == g2.get_winner()
        assert g1.serialize()["action_log"] == g2.serialize()["action_log"]


# ── factory ────────────────────────────────────────────────────────────────


class TestBaselineFactory:
    def test_create_baseline_agent(self):
        from games.avalon import create_baseline_agent

        agent = create_baseline_agent("test_player", seed=0)
        assert agent.agent_id == "test_player"
        assert agent.agent_type.value == "rule_based"
