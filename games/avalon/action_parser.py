"""Action parser for Avalon – extracts valid game actions from LLM output.

Parsing strategy (applied in order until one succeeds):
1. Extract JSON  ->  normalise fields  ->  validate against valid_actions.
2. Natural-language heuristics (regex) for each action type.
3. Random-action fallback (when ``fallback_to_random`` is enabled).
"""

import re
from typing import Any, Dict, List, Optional

from core.action_parser import ActionParser, ActionParseError


class AvalonActionParser(ActionParser):
    """Parses LLM output into one of the four Avalon action types."""

    def parse(
        self,
        raw_output: str,
        valid_actions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        # ── 1. JSON path ────────────────────────────────────────────────
        parsed = self.extract_json(raw_output)
        if parsed:
            normalised = self._normalise(parsed, valid_actions)
            if normalised:
                return normalised

        # ── 2. natural-language fallback ──────────────────────────────
        nl = self._parse_natural_language(raw_output, valid_actions)
        if nl:
            return nl

        # ── 3. random fallback ────────────────────────────────────────
        if self.fallback_to_random and valid_actions:
            return self.get_random_action(valid_actions)

        raise ActionParseError(
            "Could not extract a valid Avalon action", raw_output=raw_output[:500]
        )

    # ------------------------------------------------------------------
    # normalisation  (JSON -> canonical action dict)
    # ------------------------------------------------------------------

    def _normalise(
        self, parsed: Dict[str, Any], valid_actions: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Map the parsed JSON dict onto a well-formed action, or None."""
        action_type = str(parsed.get("action_type", "")).upper()

        # ── PROPOSE_TEAM ──────────────────────────────────────────────
        if action_type == "PROPOSE_TEAM":
            team = parsed.get("team")
            if isinstance(team, list):
                try:
                    team = [int(t) for t in team]
                except (ValueError, TypeError):
                    return None
                return {"action_type": "PROPOSE_TEAM", "team": team}

        # ── VOTE ──────────────────────────────────────────────────────
        if action_type == "VOTE":
            approve = parsed.get("approve")
            if approve is None:
                approve = parsed.get("decision")
            if isinstance(approve, bool):
                return {"action_type": "VOTE", "approve": approve}
            if isinstance(approve, str):
                low = approve.strip().lower()
                if low in ("true", "yes", "approve", "1"):
                    return {"action_type": "VOTE", "approve": True}
                if low in ("false", "no", "reject", "0"):
                    return {"action_type": "VOTE", "approve": False}

        # ── PLAY_MISSION_CARD ─────────────────────────────────────────
        if action_type == "PLAY_MISSION_CARD":
            card = str(parsed.get("card", "")).strip().lower()
            if card == "success":
                return {"action_type": "PLAY_MISSION_CARD", "card": "Success"}
            if card == "fail":
                # Only allow Fail if it appears in the valid-action list
                # (Resistance players are restricted to Success by the engine).
                if any(va.get("card") == "Fail" for va in valid_actions):
                    return {"action_type": "PLAY_MISSION_CARD", "card": "Fail"}
                return {"action_type": "PLAY_MISSION_CARD", "card": "Success"}

        # ── ASSASSINATE ───────────────────────────────────────────────
        if action_type == "ASSASSINATE":
            target = parsed.get("target")
            if target is not None:
                try:
                    return {"action_type": "ASSASSINATE", "target": int(target)}
                except (ValueError, TypeError):
                    return None

        return None

    # ------------------------------------------------------------------
    # natural-language heuristics
    # ------------------------------------------------------------------

    def _parse_natural_language(
        self, text: str, valid_actions: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not valid_actions:
            return None

        first_type = valid_actions[0].get("action_type", "")
        text_lower = text.lower()

        if first_type == "PROPOSE_TEAM":
            return self._nl_propose(text_lower)
        if first_type == "VOTE":
            return self._nl_vote(text_lower)
        if first_type == "PLAY_MISSION_CARD":
            return self._nl_mission(text_lower, valid_actions)
        if first_type == "ASSASSINATE":
            return self._nl_assassinate(text_lower, valid_actions)

        return None

    # ── per-type NL helpers ─────────────────────────────────────────────

    def _nl_propose(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract a team list from natural language."""
        patterns = [
            r'(?:team|send|propose|players?)[:\s]+\[?([\d,\s]+)\]?',
            r'\b([\d]+(?:\s*[,\s]\s*[\d]+)+)\b',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                nums = re.findall(r'\d+', match.group(1))
                if nums:
                    return {"action_type": "PROPOSE_TEAM", "team": [int(n) for n in nums]}
        return None

    def _nl_vote(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract approve / reject from natural language."""
        approve_phrases = [
            "i approve", "approve", "i support", "vote yes",
            "i vote yes", "approve the team", "i agree", "yes",
        ]
        reject_phrases = [
            "i reject", "reject", "i do not approve", "vote no",
            "i vote no", "reject the team", "i disagree",
            "i don't approve", "i will not approve", "no",
        ]
        # Check reject first – "no" is a substring of many approve phrases;
        # longer reject phrases take priority.
        for phrase in reject_phrases:
            if phrase in text:
                return {"action_type": "VOTE", "approve": False}
        for phrase in approve_phrases:
            if phrase in text:
                return {"action_type": "VOTE", "approve": True}
        return None

    def _nl_mission(
        self, text: str, valid_actions: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Extract Success / Fail from natural language."""
        fail_available = any(va.get("card") == "Fail" for va in valid_actions)
        if "fail" in text and fail_available:
            return {"action_type": "PLAY_MISSION_CARD", "card": "Fail"}
        # Default to Success (always legal; Spies must explicitly say "fail").
        return {"action_type": "PLAY_MISSION_CARD", "card": "Success"}

    def _nl_assassinate(
        self, text: str, valid_actions: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Extract assassination target from natural language."""
        valid_targets = valid_actions[0].get("valid_targets", []) if valid_actions else []

        patterns = [
            r'(?:target|assassinate|kill|choose|pick|guess)\s*(?:player\s*)?(\d+)',
            r'(?:i\s+(?:think|believe|guess)\s+(?:it\s+is|merlin\s+is)\s+)(?:player\s+)?(\d+)',
            r'\bplayer\s+(\d+)\b',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                target = int(match.group(1))
                if target in valid_targets:
                    return {"action_type": "ASSASSINATE", "target": target}

        # Last resort: first valid-target number that appears anywhere in text.
        for target in valid_targets:
            if str(target) in text:
                return {"action_type": "ASSASSINATE", "target": target}

        return None
