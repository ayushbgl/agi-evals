"""
Abstract Agent interface for game-playing agents.

Agents encapsulate decision-making logic, whether from LLMs,
rule-based systems, or human input.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from enum import Enum


class AgentType(Enum):
    """Types of agents that can play games."""
    LLM = "llm"
    MANUAL = "manual"
    RANDOM = "random"
    RULE_BASED = "rule_based"


class Agent(ABC):
    """
    Abstract base class for game-playing agents.

    An agent receives game state information and returns actions.
    Implementations include:
    - LLMAgent: Queries an LLM for decisions
    - ManualAgent: Pauses for human input
    - RandomAgent: Selects random valid actions
    - RuleBasedAgent: Uses programmatic strategies

    Agents are stateless with respect to the game - all information
    needed to make a decision is passed via the decide() method.

    Example:
        class MyAgent(Agent):
            def decide(self, prompt, valid_actions, **kwargs):
                # Custom decision logic
                return valid_actions[0]
    """

    def __init__(
        self,
        agent_id: str,
        agent_type: AgentType,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the agent.

        Args:
            agent_id: Unique identifier for this agent
            agent_type: Type of agent (LLM, MANUAL, etc.)
            config: Optional configuration dict
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config or {}

    @abstractmethod
    def decide(
        self,
        prompt: str,
        valid_actions: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Make a decision given the current game state.

        Args:
            prompt: Formatted prompt describing the game state
            valid_actions: List of valid actions to choose from
            system_prompt: Optional system prompt for LLM agents
            **kwargs: Additional context (turn_history, metadata, etc.)

        Returns:
            Dict containing:
                - action: The chosen action dict
                - reasoning: Optional explanation (for logging)
                - raw_output: Optional raw response (for LLMs)
                - metadata: Optional additional data (tokens, latency, etc.)
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get agent metadata for logging.

        Returns:
            Dict with agent info (id, type, model name, etc.)
        """
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
        }


class RandomAgent(Agent):
    """
    Agent that selects random valid actions.

    Useful for testing and as a baseline opponent.
    """

    def __init__(self, agent_id: str, seed: Optional[int] = None):
        super().__init__(agent_id, AgentType.RANDOM)
        self._seed = seed
        if seed is not None:
            import random
            self._rng = random.Random(seed)
        else:
            import random
            self._rng = random.Random()

    def decide(
        self,
        prompt: str,
        valid_actions: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Select a random valid action."""
        if not valid_actions:
            raise ValueError("No valid actions available")

        action = self._rng.choice(valid_actions)
        return {
            "action": action,
            "reasoning": "Random selection",
            "raw_output": "",
            "metadata": {"method": "random"},
        }


class ManualAgent(Agent):
    """
    Agent that pauses for human input via console.

    Useful for manual testing and debugging. Prints the prompt
    and waits for the user to paste a JSON response.
    """

    def __init__(
        self,
        agent_id: str,
        output_file: Optional[str] = None,
    ):
        """
        Initialize ManualAgent.

        Args:
            agent_id: Unique identifier
            output_file: Optional file to write prompts to
        """
        super().__init__(agent_id, AgentType.MANUAL)
        self.output_file = output_file

    def decide(
        self,
        prompt: str,
        valid_actions: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Display prompt and wait for user input.

        The user should paste a JSON response with the action.
        """
        import json

        separator = "=" * 60
        print(f"\n{separator}")
        print("MANUAL AGENT INPUT REQUIRED")
        print(separator)

        if system_prompt:
            print("\n[SYSTEM PROMPT]")
            print(system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt)

        print("\n[GAME STATE PROMPT]")
        print(prompt)

        print("\n[VALID ACTIONS]")
        for i, action in enumerate(valid_actions[:10]):  # Show first 10
            print(f"  {i}: {action}")
        if len(valid_actions) > 10:
            print(f"  ... and {len(valid_actions) - 10} more")

        print(f"\n{separator}")

        # Write to file if specified
        if self.output_file:
            with open(self.output_file, 'w') as f:
                json.dump({
                    "system_prompt": system_prompt,
                    "prompt": prompt,
                    "valid_actions": valid_actions,
                }, f, indent=2)
            print(f"Prompt written to: {self.output_file}")

        print("\nPaste your JSON response (or action index number):")
        print("Example: {\"action_type\": \"...\", \"value\": ...}")
        print("Or just type a number to select from valid actions above.")

        while True:
            try:
                user_input = input("> ").strip()

                # Check if it's just a number (action index)
                if user_input.isdigit():
                    idx = int(user_input)
                    if 0 <= idx < len(valid_actions):
                        action = valid_actions[idx]
                        return {
                            "action": action,
                            "reasoning": "Manual selection by index",
                            "raw_output": user_input,
                            "metadata": {"method": "index_selection"},
                        }
                    else:
                        print(f"Invalid index. Must be 0-{len(valid_actions)-1}")
                        continue

                # Try to parse as JSON
                data = json.loads(user_input)

                # If data has "action" key, extract it
                if "action" in data:
                    action = data["action"]
                else:
                    action = data

                return {
                    "action": action,
                    "reasoning": data.get("reasoning", "Manual input"),
                    "raw_output": user_input,
                    "metadata": {"method": "json_input"},
                }

            except json.JSONDecodeError:
                print("Invalid JSON. Please try again.")
            except KeyboardInterrupt:
                print("\nUsing first valid action as fallback.")
                return {
                    "action": valid_actions[0] if valid_actions else {},
                    "reasoning": "Keyboard interrupt - fallback",
                    "raw_output": "",
                    "metadata": {"method": "interrupt_fallback"},
                }
