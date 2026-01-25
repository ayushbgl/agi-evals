"""
LLM Agent - Makes decisions using LLM API calls.

Supports multiple providers: OpenAI, Anthropic, Google, etc.
"""

import json
import os
import time
from typing import Dict, Any, List, Optional
from core.agent import Agent, AgentType


class LLMAgent(Agent):
    """
    Agent that uses LLM APIs for decision-making.

    Supports:
    - OpenAI (GPT-4, GPT-4o, etc.)
    - Anthropic (Claude 3)
    - Google (Gemini)
    """

    def __init__(
        self,
        agent_id: str,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        super().__init__(agent_id, AgentType.LLM)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None
        self._provider = self._detect_provider(model)

    def _detect_provider(self, model: str) -> str:
        """Detect which provider to use based on model name."""
        model_lower = model.lower()
        if "gpt" in model_lower or "o1" in model_lower:
            return "openai"
        elif "claude" in model_lower:
            return "anthropic"
        elif "gemini" in model_lower:
            return "google"
        elif "llama" in model_lower or "mistral" in model_lower:
            return "together"  # or other provider
        else:
            return "openai"  # default

    def _get_openai_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return self._client

    def _get_anthropic_client(self):
        """Get or create Anthropic client."""
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        return self._client

    def _get_google_client(self):
        """Get or create Google AI client."""
        if self._client is None:
            import google.generativeai as genai
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
            self._client = genai.GenerativeModel(self.model)
        return self._client

    def decide(
        self,
        prompt: str,
        valid_actions: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Query LLM for decision.

        Returns action dict with reasoning and metadata.
        """
        start_time = time.time()

        try:
            if self._provider == "openai":
                response = self._call_openai(prompt, system_prompt)
            elif self._provider == "anthropic":
                response = self._call_anthropic(prompt, system_prompt)
            elif self._provider == "google":
                response = self._call_google(prompt, system_prompt)
            else:
                raise ValueError(f"Unknown provider: {self._provider}")

            latency_ms = int((time.time() - start_time) * 1000)

            # Parse response
            raw_output = response["content"]
            parsed = self._parse_response(raw_output, valid_actions)

            return {
                "action": parsed["action"],
                "reasoning": parsed.get("reasoning", ""),
                "raw_output": raw_output,
                "metadata": {
                    "model": self.model,
                    "provider": self._provider,
                    "latency_ms": latency_ms,
                    "prompt_tokens": response.get("prompt_tokens", 0),
                    "completion_tokens": response.get("completion_tokens", 0),
                    "total_tokens": response.get("total_tokens", 0),
                },
            }

        except Exception as e:
            # On error, fall back to first valid action
            latency_ms = int((time.time() - start_time) * 1000)
            return {
                "action": valid_actions[0] if valid_actions else {},
                "reasoning": f"LLM error: {e}",
                "raw_output": str(e),
                "metadata": {
                    "model": self.model,
                    "provider": self._provider,
                    "latency_ms": latency_ms,
                    "error": str(e),
                },
            }

    def _call_openai(self, prompt: str, system_prompt: Optional[str]) -> Dict[str, Any]:
        """Call OpenAI API."""
        client = self._get_openai_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return {
            "content": response.choices[0].message.content,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    def _call_anthropic(self, prompt: str, system_prompt: Optional[str]) -> Dict[str, Any]:
        """Call Anthropic API."""
        client = self._get_anthropic_client()

        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = client.messages.create(**kwargs)

        return {
            "content": response.content[0].text,
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }

    def _call_google(self, prompt: str, system_prompt: Optional[str]) -> Dict[str, Any]:
        """Call Google AI API."""
        client = self._get_google_client()

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        response = client.generate_content(full_prompt)

        return {
            "content": response.text,
            "prompt_tokens": 0,  # Gemini doesn't report tokens the same way
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def _parse_response(
        self,
        raw_output: str,
        valid_actions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Parse LLM response into an action.

        Handles:
        - JSON responses
        - Free-text responses with action keywords
        """
        # Try to extract JSON from response
        try:
            # Look for JSON in the response
            import re
            json_match = re.search(r'\{[^{}]*\}', raw_output, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                # Extract action and reasoning
                action = data.get("action", data)
                reasoning = data.get("reasoning", "")

                # Map to valid action if needed
                mapped = self._map_to_valid_action(action, valid_actions)
                return {"action": mapped, "reasoning": reasoning}
        except (json.JSONDecodeError, AttributeError):
            pass

        # Try to match keywords in response
        upper = raw_output.upper()

        # Check for action types
        for va in valid_actions:
            action_type = va.get("action_type", "")
            if action_type in upper:
                # Try to extract value
                if action_type == "GUESS":
                    # Look for word
                    word = va.get("word", "")
                    if word and word.upper() in upper:
                        return {"action": va, "reasoning": f"Matched {word}"}
                elif action_type == "GIVE_CLUE":
                    # Extract clue word and number
                    return {"action": self._extract_clue(raw_output), "reasoning": "Extracted clue"}
                elif action_type == "PASS":
                    return {"action": va, "reasoning": "Pass detected"}

        # Fallback: return first valid action
        return {
            "action": valid_actions[0] if valid_actions else {},
            "reasoning": "Could not parse - using fallback",
        }

    def _map_to_valid_action(
        self,
        action: Dict[str, Any],
        valid_actions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Map parsed action to a valid action."""
        action_type = action.get("action_type", "")

        for va in valid_actions:
            if va.get("action_type") == action_type:
                # For GUESS, match word
                if action_type == "GUESS":
                    if va.get("word", "").upper() == action.get("word", "").upper():
                        return va
                # For other types, return if matches
                elif action_type in ["PASS", "GIVE_CLUE"]:
                    return {**va, **action}

        # If no match found, return original
        return action

    def _extract_clue(self, text: str) -> Dict[str, Any]:
        """Extract clue word and number from text."""
        import re

        # Look for patterns like "CLUE: word 3" or "word for 3"
        patterns = [
            r'["\']?(\w+)["\']?\s*(?:for|:)?\s*(\d+)',
            r'clue[:\s]+["\']?(\w+)["\']?\s*,?\s*(\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return {
                    "action_type": "GIVE_CLUE",
                    "clue": match.group(1).upper(),
                    "number": int(match.group(2)),
                }

        # Fallback - just return a generic clue
        words = [w for w in text.split() if w.isalpha() and len(w) > 2]
        return {
            "action_type": "GIVE_CLUE",
            "clue": words[0].upper() if words else "HINT",
            "number": 1,
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Get agent metadata."""
        return {
            "agent_id": self.agent_id,
            "agent_type": "llm",
            "model": self.model,
            "provider": self._provider,
            "temperature": self.temperature,
        }
