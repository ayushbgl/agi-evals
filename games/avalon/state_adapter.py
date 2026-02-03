"""State adapter for Avalon – converts game state to LLM prompts.

The adapter assembles role-specific system prompts from ``prompts.py``
and builds the user message from the live public + private state.
"""

from typing import Any, Dict, List, Optional

from core.state_adapter import StateAdapter
from games.avalon.prompts import (
    ASSASSINATION_INSTRUCTION,
    MISSION_CARD_INSTRUCTION_RESISTANCE,
    MISSION_CARD_INSTRUCTION_SPY,
    PROPOSE_TEAM_INSTRUCTION,
    ROLE_SYSTEM_PROMPTS,
    VOTE_INSTRUCTION,
)


class AvalonStateAdapter(StateAdapter):
    """Converts Avalon game state into role- and phase-specific LLM prompts."""

    # ------------------------------------------------------------------
    # system prompt
    # ------------------------------------------------------------------

    def format_system_prompt(self, role: Optional[str] = None, **kwargs) -> str:
        return ROLE_SYSTEM_PROMPTS.get(
            role or "",
            "You are playing The Resistance: Avalon. Play strategically.",
        )

    # ------------------------------------------------------------------
    # user prompt  (main entry point)
    # ------------------------------------------------------------------

    def state_to_prompt(
        self,
        public_state: Dict[str, Any],
        private_state: Dict[str, Any],
        valid_actions: List[Dict[str, Any]],
        turn_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        sections: List[str] = []

        sections.append(self._format_identity(private_state))
        sections.append(self._format_game_status(public_state))

        knowledge_text = self._format_knowledge(private_state)
        if knowledge_text:
            sections.append(knowledge_text)

        history_text = self._format_round_history(public_state)
        if history_text:
            sections.append(history_text)

        round_ctx = self._format_current_round(public_state)
        if round_ctx:
            sections.append(round_ctx)

        sections.append(self._format_action_instruction(public_state, private_state))

        return "\n".join(sections)

    # ------------------------------------------------------------------
    # output schema
    # ------------------------------------------------------------------

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "oneOf": [
                {
                    "type": "object",
                    "description": "Propose a mission team (Leader only)",
                    "required": ["action_type", "team"],
                    "properties": {
                        "action_type": {"type": "string", "enum": ["PROPOSE_TEAM"]},
                        "team": {"type": "array", "items": {"type": "integer"}},
                        "reasoning": {"type": "string"},
                    },
                },
                {
                    "type": "object",
                    "description": "Vote on a team proposal",
                    "required": ["action_type", "approve"],
                    "properties": {
                        "action_type": {"type": "string", "enum": ["VOTE"]},
                        "approve": {"type": "boolean"},
                        "reasoning": {"type": "string"},
                    },
                },
                {
                    "type": "object",
                    "description": "Play a mission card",
                    "required": ["action_type", "card"],
                    "properties": {
                        "action_type": {"type": "string", "enum": ["PLAY_MISSION_CARD"]},
                        "card": {"type": "string", "enum": ["Success", "Fail"]},
                        "reasoning": {"type": "string"},
                    },
                },
                {
                    "type": "object",
                    "description": "Assassinate – guess who is Merlin",
                    "required": ["action_type", "target"],
                    "properties": {
                        "action_type": {"type": "string", "enum": ["ASSASSINATE"]},
                        "target": {"type": "integer"},
                        "reasoning": {"type": "string"},
                    },
                },
            ]
        }

    # ------------------------------------------------------------------
    # section builders
    # ------------------------------------------------------------------

    def _format_identity(self, private_state: Dict[str, Any]) -> str:
        role = private_state.get("role", "Unknown")
        alignment = private_state.get("alignment", "unknown")
        idx = private_state.get("player_index", "?")
        return (
            f"# The Resistance: Avalon\n\n"
            f"**You are Player {idx}** | Role: **{role}** | Alignment: **{alignment.title()}**\n"
        )

    def _format_game_status(self, pub: Dict[str, Any]) -> str:
        n = pub.get("num_players", "?")
        mission = pub.get("mission_number", "?")
        rw = pub.get("resistance_wins", 0)
        sw = pub.get("spy_wins", 0)
        leader = pub.get("current_leader", "?")
        sizes = pub.get("mission_sizes", [])
        two_fail = pub.get("requires_two_fails_mission4", False)

        # Mission-size bar: e.g.  "X  O  [2]  3  3"
        results = pub.get("mission_results", [])
        icons = []
        for i in range(5):
            if i < len(results):
                icons.append("X" if results[i]["mission_failed"] else "O")
            elif i + 1 == mission:
                icons.append(f"[{sizes[i]}]" if i < len(sizes) else "[?]")
            else:
                icons.append(str(sizes[i]) if i < len(sizes) else "?")
        mission_bar = "  ".join(icons)

        lines = [
            "## Game Status",
            f"- **Players:** {n}  |  **Current Mission:** {mission}/5",
            f"- **Score:** Resistance {rw} \u2013 {sw} Spies",
            f"- **Current Leader:** {leader}",
            f"- **Mission sizes (O=success, X=fail, [n]=current):** {mission_bar}",
        ]
        if two_fail:
            lines.append("- Note: Mission 4 requires **two** Fail cards to fail.")

        rej = pub.get("consecutive_rejections", 0)
        if rej > 0:
            force_note = " The next proposal will be **force-approved**." if rej >= 4 else ""
            lines.append(f"- {rej} consecutive rejection(s).{force_note}")

        return "\n".join(lines) + "\n"

    def _format_knowledge(self, priv: Dict[str, Any]) -> str:
        knowledge = priv.get("knowledge", {})
        if not knowledge:
            return ""

        lines = ["## What You Know (Night Phase)"]

        suspects = [pid for pid, lbl in knowledge.items() if lbl == "suspect"]
        possible = [pid for pid, lbl in knowledge.items() if lbl == "possible_merlin"]
        allies   = [pid for pid, lbl in knowledge.items() if lbl == "fellow_spy"]

        if suspects:
            lines.append(f"- **Suspected evil:** {', '.join(suspects)}")
        if possible:
            lines.append(
                f"- **One is Merlin, the other is Morgana (indistinguishable):** "
                f"{', '.join(possible)}"
            )
        if allies:
            lines.append(f"- **Your fellow Spies:** {', '.join(allies)}")

        return "\n".join(lines) + "\n"

    def _format_round_history(self, pub: Dict[str, Any]) -> str:
        results = pub.get("mission_results", [])
        if not results:
            return ""

        lines = ["## Completed Missions"]
        for r in results:
            status = "FAILED" if r["mission_failed"] else "SUCCESS"
            team_str = ", ".join(str(t) for t in r.get("team_indices", r.get("team", [])))
            extra = " (2 Fails required)" if r.get("two_fails_required") else ""
            lines.append(
                f"- Mission {r['mission_number']}: **{status}** "
                f"| Team: [{team_str}] | Fails: {r['fail_count']}{extra}"
            )
        return "\n".join(lines) + "\n"

    def _format_current_round(self, pub: Dict[str, Any]) -> str:
        proposal_idx = pub.get("current_proposal_indices", None)
        votes = pub.get("votes", None)

        lines: List[str] = []

        if proposal_idx:
            team_display = ", ".join(str(i) for i in proposal_idx)
            lines.append("## Current Proposal")
            lines.append(f"- **Proposed team:** [{team_display}]")

        if votes:
            approve_n = sum(1 for v in votes.values() if v)
            reject_n = len(votes) - approve_n
            lines.append(f"- **Votes cast so far:** {approve_n} approve, {reject_n} reject")
            lines.append(f"- **Already voted:** {', '.join(votes.keys())}")

        return "\n".join(lines) + "\n" if lines else ""

    def _format_action_instruction(
        self, pub: Dict[str, Any], priv: Dict[str, Any]
    ) -> str:
        phase = pub.get("phase", "")
        mission = pub.get("mission_number", 1)
        proposal_idx = pub.get("current_proposal_indices", [])
        players = pub.get("players", [])

        if phase == "team_proposal":
            sizes = pub.get("mission_sizes", [])
            team_size = sizes[mission - 1] if mission <= len(sizes) else 0
            return PROPOSE_TEAM_INSTRUCTION.format(
                team_size=team_size, mission_number=mission
            )

        if phase == "team_vote":
            team_display = ", ".join(str(i) for i in proposal_idx) if proposal_idx else "?"
            return VOTE_INSTRUCTION.format(
                team_display=team_display, mission_number=mission
            )

        if phase == "mission":
            alignment = priv.get("alignment", "resistance")
            if alignment == "spy":
                note = ""
                if pub.get("requires_two_fails_mission4") and mission == 4:
                    note = "Note: This mission (Mission 4) requires **two** Fail cards to fail."
                return MISSION_CARD_INSTRUCTION_SPY.format(
                    mission_number=mission, two_fails_note=note
                )
            return MISSION_CARD_INSTRUCTION_RESISTANCE.format(mission_number=mission)

        if phase == "assassination":
            player_list = "\n".join(
                f"  - Index {i}: {pid}" for i, pid in enumerate(players)
            )
            return ASSASSINATION_INSTRUCTION + f"\n**Players:**\n{player_list}\n"

        return "The game is over.\n"

    # ------------------------------------------------------------------
    # turn history (override default)
    # ------------------------------------------------------------------

    def format_turn_history(
        self, turn_history: List[Dict[str, Any]], max_turns: int = 10
    ) -> str:
        if not turn_history:
            return "No previous actions."
        lines = []
        for entry in turn_history[-max_turns:]:
            player = entry.get("player", "?")
            action = entry.get("action", {})
            atype = action.get("action_type", "?")
            lines.append(f"- {player}: {atype}")
        return "\n".join(lines)
