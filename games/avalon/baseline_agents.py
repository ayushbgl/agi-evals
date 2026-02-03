"""Rule-based baseline agents for Avalon.

A single ``AvalonBaselineAgent`` class whose strategy adapts at
decide-time based on the role reported in ``private_state``.  No
external models, no prompt parsing — decisions are made purely from
the structured state dicts that the runner passes through kwargs.

Heuristic summary
-----------------
PROPOSE_TEAM
  Resistance / Percival
      Avoid players who appeared on any previously failed mission
      ("tainted").  Fill remaining slots randomly.
  Merlin
      Same as above, but also avoids players tagged ``"suspect"`` in
      night-phase knowledge (Assassin, Minion).
  Spies (Morgana / Assassin / Minion)
      Must-include self (guarantees a Fail card can be played).
      Fill remaining slots from players who are *not* fellow spies
      (avoids clustering spies on one team).

VOTE
  Resistance / Percival
      Reject if any member of the proposed team is tainted;
      approve otherwise.
  Merlin
      Reject if any member is a known suspect; if not, fall through
      to the tainted check above.
  Spies
      Approve if at least one fellow spy (or self) is on the proposed
      team; reject otherwise.

PLAY_MISSION_CARD
  Resistance  – Success  (only legal card).
  Spies       – Fail     (aggressive baseline).

ASSASSINATE  (Assassin only)
  Known fellow spies cannot be Merlin — filter them out of the
  valid-target list and pick uniformly at random from the rest.
"""

import random
from typing import Any, Dict, List, Optional, Set, Tuple

from core.agent import Agent, AgentType


class AvalonBaselineAgent(Agent):
    """Role-adaptive rule-based agent for Avalon.

    Instantiate once per player.  The correct strategy is selected
    automatically on every ``decide`` call based on the ``role`` field
    in ``private_state``.

    Args:
        agent_id: Unique player identifier.
        seed:     Optional RNG seed for deterministic team / target
                  selection.
    """

    _SPY_ROLES: Set[str] = {"Morgana", "Assassin", "Minion"}

    def __init__(self, agent_id: str, seed: Optional[int] = None):
        super().__init__(agent_id, AgentType.RULE_BASED)
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def decide(
        self,
        prompt: str,
        valid_actions: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        pub: Dict[str, Any] = kwargs.get("public_state", {})
        priv: Dict[str, Any] = kwargs.get("private_state", {})

        action_type = valid_actions[0]["action_type"] if valid_actions else ""

        if action_type == "PROPOSE_TEAM":
            action, reason = self._propose(pub, priv, valid_actions)
        elif action_type == "VOTE":
            action, reason = self._vote(pub, priv)
        elif action_type == "PLAY_MISSION_CARD":
            action, reason = self._mission_card(priv)
        elif action_type == "ASSASSINATE":
            action, reason = self._assassinate(pub, priv, valid_actions)
        else:
            action = valid_actions[0] if valid_actions else {}
            reason = "fallback: unknown action type"

        return {
            "action": action,
            "reasoning": reason,
            "raw_output": "",
            "metadata": {"method": "rule_based", "role": priv.get("role", "unknown")},
        }

    # ------------------------------------------------------------------
    # phase handlers
    # ------------------------------------------------------------------

    def _propose(
        self,
        pub: Dict[str, Any],
        priv: Dict[str, Any],
        valid_actions: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], str]:
        role = priv.get("role", "")
        team_size: int = valid_actions[0].get("team_size", 0)
        n: int = pub.get("num_players", 0)
        tainted = _tainted_indices(pub)

        if role == "Merlin":
            avoid = tainted | _knowledge_indices(pub, priv, "suspect")
            team = self._build_team(n, team_size, must_include=[], avoid=avoid)
            return (
                {"action_type": "PROPOSE_TEAM", "team": team},
                f"Merlin: avoiding {len(avoid)} suspect/tainted indices",
            )

        if role in self._SPY_ROLES:
            self_idx = priv.get("player_index", 0)
            other_spies = _knowledge_indices(pub, priv, "fellow_spy")
            team = self._build_team(n, team_size, must_include=[self_idx], avoid=other_spies)
            return (
                {"action_type": "PROPOSE_TEAM", "team": team},
                "Spy: including self, hiding other spies",
            )

        # Resistance / Percival / Loyal Servant
        team = self._build_team(n, team_size, must_include=[], avoid=tainted)
        return (
            {"action_type": "PROPOSE_TEAM", "team": team},
            f"Resistance: avoiding {len(tainted)} tainted indices",
        )

    def _vote(
        self,
        pub: Dict[str, Any],
        priv: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str]:
        role = priv.get("role", "")
        team_on_table = set(pub.get("current_proposal_indices") or [])

        # Merlin: reject on known suspects first, then fall through to
        # the generic tainted check at the bottom.
        if role == "Merlin":
            if team_on_table & _knowledge_indices(pub, priv, "suspect"):
                return (
                    {"action_type": "VOTE", "approve": False},
                    "Merlin: known suspect on proposed team",
                )

        # Spies: approve iff at least one spy is on the team.
        if role in self._SPY_ROLES:
            spy_indices = _knowledge_indices(pub, priv, "fellow_spy")
            spy_indices.add(priv.get("player_index", -1))  # self is also a spy
            if team_on_table & spy_indices:
                return (
                    {"action_type": "VOTE", "approve": True},
                    "Spy: a spy is on the team",
                )
            return (
                {"action_type": "VOTE", "approve": False},
                "Spy: no spy on the team",
            )

        # Generic Resistance / Percival / Merlin tainted-fallthrough
        if team_on_table & _tainted_indices(pub):
            return (
                {"action_type": "VOTE", "approve": False},
                "Resistance: tainted player on proposed team",
            )
        return (
            {"action_type": "VOTE", "approve": True},
            "Resistance: no tainted players on team",
        )

    @staticmethod
    def _mission_card(priv: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        if priv.get("alignment") == "spy":
            return (
                {"action_type": "PLAY_MISSION_CARD", "card": "Fail"},
                "Spy: play Fail",
            )
        return (
            {"action_type": "PLAY_MISSION_CARD", "card": "Success"},
            "Resistance: play Success",
        )

    def _assassinate(
        self,
        pub: Dict[str, Any],
        priv: Dict[str, Any],
        valid_actions: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], str]:
        valid_targets: List[int] = valid_actions[0].get("valid_targets", [])
        spy_indices = _knowledge_indices(pub, priv, "fellow_spy")
        # Spies cannot be Merlin; filter them out.
        candidates = [t for t in valid_targets if t not in spy_indices]
        if not candidates:
            candidates = valid_targets  # fallback: pick from everyone
        target = self._rng.choice(candidates)
        return (
            {"action_type": "ASSASSINATE", "target": target},
            f"Assassin: filtered {len(spy_indices)} known spies, {len(candidates)} candidates remain",
        )

    # ------------------------------------------------------------------
    # team builder
    # ------------------------------------------------------------------

    def _build_team(
        self,
        num_players: int,
        team_size: int,
        must_include: List[int],
        avoid: Set[int],
    ) -> List[int]:
        """Assemble a team with soft avoidance.

        Priority:
          1. ``must_include`` indices added first (capped at team_size).
          2. Remaining slots filled randomly from indices not in *avoid*.
          3. If still short, fill from *avoid* (constraint is soft —
             never produces a team smaller than requested).
        """
        picked: List[int] = []
        seen: Set[int] = set()

        for idx in must_include:
            if len(picked) >= team_size:
                break
            if idx not in seen:
                picked.append(idx)
                seen.add(idx)

        # Pool 1: non-avoided, non-picked
        pool = [i for i in range(num_players) if i not in seen and i not in avoid]
        self._rng.shuffle(pool)
        while len(picked) < team_size and pool:
            picked.append(pool.pop())
            seen.add(picked[-1])

        # Pool 2: avoided indices (last resort)
        if len(picked) < team_size:
            pool = [i for i in range(num_players) if i not in seen]
            self._rng.shuffle(pool)
            while len(picked) < team_size and pool:
                picked.append(pool.pop())

        return picked


# ── module-level helpers (stateless, no RNG) ───────────────────────────────


def _tainted_indices(pub: Dict[str, Any]) -> Set[int]:
    """Indices of players who appeared on any previously failed mission."""
    tainted: Set[int] = set()
    for m in pub.get("mission_results", []):
        if m.get("mission_failed"):
            tainted.update(m.get("team_indices", []))
    return tainted


def _knowledge_indices(
    pub: Dict[str, Any], priv: Dict[str, Any], label: str
) -> Set[int]:
    """Convert knowledge entries with *label* to player indices."""
    players: List[str] = pub.get("players", [])
    return {
        players.index(pid)
        for pid, lbl in priv.get("knowledge", {}).items()
        if lbl == label and pid in players
    }
