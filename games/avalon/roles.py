"""Role definitions, visibility rules, and static game tables for Avalon.

All player-count-dependent constants (role pools, team sizes, double-fail
thresholds) live here so that the engine in game.py stays table-driven.
Adding or tweaking a role requires changes only in this module.
"""

from enum import Enum
from typing import Dict, List, Set


class Role(Enum):
    """Avalon character roles."""

    MERLIN = "Merlin"
    PERCIVAL = "Percival"
    LOYAL_SERVANT = "Loyal Servant"
    MORGANA = "Morgana"
    ASSASSIN = "Assassin"
    MINION = "Minion"
    OBERON = "Oberon"


# ── alignment sets ────────────────────────────────────────────────────────────

SPY_ROLES: Set[Role] = {Role.MORGANA, Role.ASSASSIN, Role.MINION}
RESISTANCE_ROLES: Set[Role] = {Role.MERLIN, Role.PERCIVAL, Role.LOYAL_SERVANT, Role.OBERON}


def is_spy(role: Role) -> bool:
    """Return True if *role* belongs to the Spy faction."""
    return role in SPY_ROLES


def is_resistance(role: Role) -> bool:
    """Return True if *role* belongs to the Resistance faction."""
    return role in RESISTANCE_ROLES


# ── role descriptions (consumed by the prompt layer) ─────────────────────────

ROLE_DESCRIPTIONS: Dict[str, str] = {
    "Merlin": (
        "You know which players are evil – specifically the Assassin and any "
        "Minions. Morgana is hidden from you and appears innocent. "
        "If the Resistance wins three missions the Assassin will try to identify "
        "you; do not make your identity obvious."
    ),
    "Percival": (
        "You can see two players: one is Merlin (Resistance) and one is Morgana "
        "(Spy). You cannot tell them apart. Use deduction to support the real Merlin."
    ),
    "Loyal Servant": (
        "You have no special information. Deduce the truth through observation of "
        "voting patterns, team proposals, and mission outcomes."
    ),
    "Morgana": (
        "You know all fellow Spies. Percival sees you as a possible Merlin – "
        "exploit this confusion. You may play Fail cards on missions."
    ),
    "Assassin": (
        "You know all fellow Spies. If the Resistance reaches three successes you "
        "will get exactly one chance to assassinate Merlin. Choose wisely."
    ),
    "Minion": (
        "You know all fellow Spies. Help sabotage the Resistance through strategic "
        "voting and mission play."
    ),
    "Oberon": (
        "You are Resistance but the Spies mistakenly believe you are one of them. "
        "You can see the actual evil players. Be careful – acting too knowledgeable "
        "may reveal that you are not a Spy."
    ),
}


# ── default role pools (keyed by player count) ───────────────────────────────

DEFAULT_ROLE_POOL: Dict[int, List[Role]] = {
    5: [
        Role.MERLIN, Role.PERCIVAL, Role.LOYAL_SERVANT,
        Role.MORGANA, Role.ASSASSIN,
    ],
    6: [
        Role.MERLIN, Role.PERCIVAL, Role.LOYAL_SERVANT, Role.LOYAL_SERVANT,
        Role.MORGANA, Role.ASSASSIN,
    ],
    7: [
        Role.MERLIN, Role.PERCIVAL, Role.LOYAL_SERVANT, Role.LOYAL_SERVANT,
        Role.MORGANA, Role.ASSASSIN, Role.MINION,
    ],
    8: [
        Role.MERLIN, Role.PERCIVAL,
        Role.LOYAL_SERVANT, Role.LOYAL_SERVANT, Role.LOYAL_SERVANT,
        Role.MORGANA, Role.ASSASSIN, Role.MINION,
    ],
    9: [
        Role.MERLIN, Role.PERCIVAL,
        Role.LOYAL_SERVANT, Role.LOYAL_SERVANT, Role.LOYAL_SERVANT, Role.LOYAL_SERVANT,
        Role.MORGANA, Role.ASSASSIN, Role.MINION,
    ],
    10: [
        Role.MERLIN, Role.PERCIVAL,
        Role.LOYAL_SERVANT, Role.LOYAL_SERVANT, Role.LOYAL_SERVANT, Role.LOYAL_SERVANT,
        Role.MORGANA, Role.ASSASSIN, Role.MINION, Role.MINION,
    ],
}


# ── mission tables ────────────────────────────────────────────────────────────
# [M1, M2, M3, M4, M5] team sizes, indexed by player count.

MISSION_SIZES: Dict[int, List[int]] = {
    5:  [2, 3, 2, 3, 3],
    6:  [2, 4, 3, 3, 4],
    7:  [2, 4, 3, 4, 5],
    8:  [3, 4, 4, 5, 5],
    9:  [3, 4, 4, 5, 5],
    10: [3, 5, 4, 6, 5],
}


def mission_team_size(num_players: int, mission_number: int) -> int:
    """Return the team size for *mission_number* (1-based, 1–5)."""
    return MISSION_SIZES[num_players][mission_number - 1]


def requires_two_fails(num_players: int, mission_number: int) -> bool:
    """True when mission 4 is played with 7 or more players."""
    return mission_number == 4 and num_players >= 7


# ── night-phase visibility ────────────────────────────────────────────────────


def compute_knowledge(
    my_id: str,
    my_role: Role,
    assignments: Dict[str, Role],
) -> Dict[str, str]:
    """
    Compute the role-specific night-phase knowledge for *my_id*.

    Returns a dict mapping *other* player IDs to one of:
        ``"suspect"``          – known evil (Merlin / Oberon view)
        ``"fellow_spy"``       – known spy ally (spy view; includes Oberon)
        ``"possible_merlin"``  – is Merlin or Morgana, indistinguishable (Percival)

    Players absent from the returned dict are unknown to this role.
    """
    knowledge: Dict[str, str] = {}

    if my_role == Role.MERLIN:
        # Merlin sees Assassin and Minion; Morgana is deliberately hidden.
        for pid, role in assignments.items():
            if pid == my_id:
                continue
            if role in (Role.ASSASSIN, Role.MINION):
                knowledge[pid] = "suspect"

    elif my_role == Role.PERCIVAL:
        # Percival sees Merlin and Morgana but cannot distinguish them.
        for pid, role in assignments.items():
            if pid == my_id:
                continue
            if role in (Role.MERLIN, Role.MORGANA):
                knowledge[pid] = "possible_merlin"

    elif my_role in SPY_ROLES:
        # All spies see each other.  Oberon is also visible (mistaken for a spy).
        for pid, role in assignments.items():
            if pid == my_id:
                continue
            if role in SPY_ROLES or role == Role.OBERON:
                knowledge[pid] = "fellow_spy"

    elif my_role == Role.OBERON:
        # Oberon sees the true evil players.
        for pid, role in assignments.items():
            if pid == my_id:
                continue
            if role in SPY_ROLES:
                knowledge[pid] = "suspect"

    # Loyal Servant: knowledge remains empty – no night-phase information.
    return knowledge
