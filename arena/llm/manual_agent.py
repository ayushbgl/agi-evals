"""
ManualAgent for human interaction via console.

Allows testing game flows by pausing for human input,
displaying prompts, and accepting JSON or index-based responses.
"""

import json
import sys
from typing import Dict, Any, List, Optional
from pathlib import Path

from core.agent import Agent, AgentType


class ManualAgent(Agent):
    """
    Agent that pauses for human input via console.

    Useful for:
    - Manual testing of game flows
    - Debugging LLM prompts
    - Human-in-the-loop experiments
    - Copy-paste workflow with external LLMs

    The agent displays the game state prompt and waits for the user
    to provide a JSON response or select an action by index.
    """

    def __init__(
        self,
        agent_id: str,
        output_file: Optional[str] = None,
        input_file: Optional[str] = None,
        interactive: bool = True,
    ):
        """
        Initialize ManualAgent.

        Args:
            agent_id: Unique identifier for this agent
            output_file: Optional file to write prompts to (for copy-paste workflow)
            input_file: Optional file to read responses from
            interactive: If False, uses file-based I/O only
        """
        super().__init__(agent_id, AgentType.MANUAL)
        self.output_file = output_file
        self.input_file = input_file
        self.interactive = interactive

    def decide(
        self,
        prompt: str,
        valid_actions: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Display prompt and wait for user input.

        Args:
            prompt: Game state prompt to display
            valid_actions: List of valid actions
            system_prompt: Optional system prompt (shown separately)
            **kwargs: Additional context

        Returns:
            Dict containing:
                - action: The chosen action
                - reasoning: User's explanation
                - raw_output: Raw user input
                - metadata: Additional info
        """
        # Display separator
        separator = "=" * 70
        print(f"\n{separator}")
        print(f"MANUAL AGENT: {self.agent_id}")
        print(separator)

        # Show context
        game_type = kwargs.get("game_type", "unknown")
        role = kwargs.get("role", "player")
        turn = kwargs.get("turn_number", "?")
        print(f"Game: {game_type} | Role: {role} | Turn: {turn}")
        print(separator)

        # Show system prompt if provided
        if system_prompt:
            print("\n[SYSTEM PROMPT]")
            # Truncate if too long
            if len(system_prompt) > 1000:
                print(system_prompt[:1000])
                print(f"... (truncated, {len(system_prompt)} chars total)")
            else:
                print(system_prompt)
            print()

        # Show game state prompt
        print("\n[GAME STATE]")
        print(prompt)

        # Show valid actions
        print("\n[VALID ACTIONS]")
        self._display_actions(valid_actions)

        # Write to file if specified
        if self.output_file:
            self._write_prompt_to_file(
                prompt, valid_actions, system_prompt, kwargs
            )

        print(f"\n{separator}")

        # Get input
        if self.input_file and Path(self.input_file).exists():
            response = self._read_from_file()
        elif self.interactive:
            response = self._get_interactive_input(valid_actions)
        else:
            raise RuntimeError(
                f"ManualAgent {self.agent_id}: No input available. "
                "Set interactive=True or provide input_file."
            )

        return response

    def _display_actions(self, valid_actions: List[Dict[str, Any]]) -> None:
        """Display valid actions with indices."""
        if not valid_actions:
            print("  (no valid actions)")
            return

        # Group by action type
        by_type: Dict[str, List[tuple]] = {}
        for i, action in enumerate(valid_actions):
            action_type = action.get("action_type", "UNKNOWN")
            if action_type not in by_type:
                by_type[action_type] = []
            by_type[action_type].append((i, action))

        for action_type, actions in by_type.items():
            if len(actions) == 1:
                i, action = actions[0]
                value = action.get("value") or action.get("word") or action.get("clue")
                if value:
                    print(f"  [{i}] {action_type}: {value}")
                else:
                    print(f"  [{i}] {action_type}")
            else:
                print(f"  {action_type}:")
                for i, action in actions[:10]:  # Show first 10
                    value = action.get("value") or action.get("word") or ""
                    print(f"    [{i}] {value}")
                if len(actions) > 10:
                    print(f"    ... and {len(actions) - 10} more")

    def _write_prompt_to_file(
        self,
        prompt: str,
        valid_actions: List[Dict[str, Any]],
        system_prompt: Optional[str],
        kwargs: Dict,
    ) -> None:
        """Write prompt data to file for copy-paste workflow."""
        output_data = {
            "agent_id": self.agent_id,
            "game_type": kwargs.get("game_type"),
            "role": kwargs.get("role"),
            "turn_number": kwargs.get("turn_number"),
            "system_prompt": system_prompt,
            "prompt": prompt,
            "valid_actions": valid_actions,
            "instructions": (
                "Respond with JSON. For action by index, use: "
                "{\"action_index\": N}. "
                "For direct action, use the action format shown."
            ),
        }

        with open(self.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nPrompt written to: {self.output_file}")

    def _read_from_file(self) -> Dict[str, Any]:
        """Read response from input file."""
        with open(self.input_file, 'r') as f:
            content = f.read().strip()

        try:
            data = json.loads(content)

            # Handle action_index format
            if "action_index" in data:
                return {
                    "action": {"_index": data["action_index"]},
                    "reasoning": data.get("reasoning", "From file"),
                    "raw_output": content,
                    "metadata": {"method": "file_index"},
                }

            # Handle direct action format
            if "action" in data:
                return {
                    "action": data["action"],
                    "reasoning": data.get("reasoning", "From file"),
                    "raw_output": content,
                    "metadata": {"method": "file_json"},
                }

            # Assume the whole thing is the action
            return {
                "action": data,
                "reasoning": data.get("reasoning", "From file"),
                "raw_output": content,
                "metadata": {"method": "file_json"},
            }

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in input file: {e}")

    def _get_interactive_input(
        self,
        valid_actions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Get input interactively from console."""
        print("\nEnter your response:")
        print("  - Type a number to select action by index")
        print("  - Paste JSON for custom response")
        print("  - Type 'q' to quit")

        while True:
            try:
                user_input = input("\n> ").strip()

                # Check for quit
                if user_input.lower() in ['q', 'quit', 'exit']:
                    print("Exiting...")
                    sys.exit(0)

                # Check if it's an index
                if user_input.isdigit():
                    idx = int(user_input)
                    if 0 <= idx < len(valid_actions):
                        action = valid_actions[idx]
                        return {
                            "action": action,
                            "reasoning": "Manual selection by index",
                            "raw_output": user_input,
                            "metadata": {"method": "index_selection", "index": idx},
                        }
                    else:
                        print(f"Invalid index. Must be 0-{len(valid_actions)-1}")
                        continue

                # Try to parse as JSON
                try:
                    data = json.loads(user_input)

                    # Handle action wrapper
                    if "action" in data:
                        action = data["action"]
                    else:
                        action = data

                    return {
                        "action": action,
                        "reasoning": data.get("reasoning", "Manual JSON input"),
                        "raw_output": user_input,
                        "metadata": {"method": "json_input"},
                    }

                except json.JSONDecodeError:
                    # Not JSON - might be a simple word/command
                    # Try to match against valid actions
                    user_upper = user_input.upper()

                    # Look for word match in guess/clue actions
                    for i, action in enumerate(valid_actions):
                        word = action.get("word", "").upper()
                        if word == user_upper:
                            return {
                                "action": action,
                                "reasoning": "Word match",
                                "raw_output": user_input,
                                "metadata": {"method": "word_match"},
                            }

                    # Check for PASS
                    if user_upper == "PASS":
                        for action in valid_actions:
                            if action.get("action_type") == "PASS":
                                return {
                                    "action": action,
                                    "reasoning": "Pass command",
                                    "raw_output": user_input,
                                    "metadata": {"method": "command"},
                                }

                    print("Could not parse input. Please enter a number, JSON, or valid word.")

            except KeyboardInterrupt:
                print("\n\nUsing first valid action as fallback.")
                if valid_actions:
                    return {
                        "action": valid_actions[0],
                        "reasoning": "Keyboard interrupt fallback",
                        "raw_output": "",
                        "metadata": {"method": "interrupt_fallback"},
                    }
                raise

            except EOFError:
                print("\nEnd of input reached.")
                if valid_actions:
                    return {
                        "action": valid_actions[0],
                        "reasoning": "EOF fallback",
                        "raw_output": "",
                        "metadata": {"method": "eof_fallback"},
                    }
                raise

    def get_metadata(self) -> Dict[str, Any]:
        """Get agent metadata."""
        return {
            "agent_id": self.agent_id,
            "agent_type": "manual",
            "interactive": self.interactive,
            "output_file": self.output_file,
            "input_file": self.input_file,
        }
