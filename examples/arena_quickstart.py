#!/usr/bin/env python3
"""
Catan-Arena Quickstart Example

This example demonstrates how to run LLM vs LLM games in Catan-Arena.

Prerequisites:
    pip install catan-arena[llm]

    Set environment variables for LLM providers:
    - ANTHROPIC_API_KEY for Claude
    - OPENAI_API_KEY for GPT-4
    - GOOGLE_API_KEY for Gemini
"""

import os
from pathlib import Path


def example_1_simple_game():
    """
    Example 1: Run a simple game with 2 LLMs and 2 random players.
    """
    print("\n" + "="*60)
    print("Example 1: Simple Game (2 LLMs + 2 Random)")
    print("="*60)

    from catan_arena.orchestration.runner import run_simple_game

    # Define players
    players = [
        {
            "id": "player_0",
            "color": "RED",
            "type": "llm",
            "model": "claude-3-haiku-20240307",  # Fast/cheap for testing
            "llm_config": {"temperature": 0.7},
        },
        {
            "id": "player_1",
            "color": "BLUE",
            "type": "random",
        },
        {
            "id": "player_2",
            "color": "ORANGE",
            "type": "random",
        },
        {
            "id": "player_3",
            "color": "WHITE",
            "type": "random",
        },
    ]

    # Run game
    result = run_simple_game(
        player_configs=players,
        map_type="BASE",
        vps_to_win=10,
        max_turns=200,  # Short game for demo
        verbose=True,
    )

    print(f"\nResult: {result}")
    return result


def example_2_llm_vs_llm():
    """
    Example 2: Claude vs GPT-4 head-to-head.
    """
    print("\n" + "="*60)
    print("Example 2: Claude vs GPT-4 (2-player)")
    print("="*60)

    from catan_arena.orchestration.runner import run_simple_game

    players = [
        {
            "id": "claude",
            "color": "RED",
            "type": "llm",
            "model": "claude-3-sonnet-20240229",
        },
        {
            "id": "gpt4",
            "color": "BLUE",
            "type": "llm",
            "model": "gpt-4-turbo",
        },
    ]

    result = run_simple_game(
        player_configs=players,
        map_type="BASE",
        vps_to_win=10,
        max_turns=300,
        verbose=True,
    )

    print(f"\nResult: {result}")
    return result


def example_3_tournament():
    """
    Example 3: Run a tournament of multiple games.
    """
    print("\n" + "="*60)
    print("Example 3: Tournament (5 games)")
    print("="*60)

    from catan_arena.orchestration.runner import run_tournament

    players = [
        {"id": "claude", "color": "RED", "type": "llm", "model": "claude-3-haiku-20240307"},
        {"id": "random_1", "color": "BLUE", "type": "random"},
        {"id": "random_2", "color": "ORANGE", "type": "random"},
        {"id": "random_3", "color": "WHITE", "type": "random"},
    ]

    result = run_tournament(
        player_configs=players,
        num_games=5,
        max_turns=200,
        verbose=False,
    )

    print(f"\nTournament Results:")
    print(f"  Wins: {result['wins']}")
    print(f"  Avg turns: {result['avg_turns']:.1f}")
    return result


def example_4_with_config():
    """
    Example 4: Using the full configuration system.
    """
    print("\n" + "="*60)
    print("Example 4: Full Configuration")
    print("="*60)

    from catan_arena.config import ArenaConfig, PlayerConfig, CatanGameConfig, LLMConfig
    from catan_arena.orchestration.runner import run_arena_game

    config = ArenaConfig(
        players=[
            PlayerConfig(
                id="claude_opus",
                color="RED",
                type="llm",
                model="claude-3-opus-20240229",
                llm_config=LLMConfig(temperature=0.5, max_tokens=2048),
            ),
            PlayerConfig(
                id="random_blue",
                color="BLUE",
                type="random",
            ),
        ],
        game_config=CatanGameConfig(
            map_type="BASE",
            vps_to_win=10,
        ),
        max_turns=300,
        log_dir="./game_logs",
        seed=42,  # Reproducible
    )

    result = run_arena_game(config, verbose=True)
    print(f"\nResult: {result}")
    return result


def example_5_replay_analysis():
    """
    Example 5: Load and analyze a game log.
    """
    print("\n" + "="*60)
    print("Example 5: Game Log Analysis")
    print("="*60)

    from catan_arena.storage.game_log import GameLogReader
    from pathlib import Path

    # Find a recent log file
    log_dir = Path("./game_logs")
    if not log_dir.exists():
        print("No game logs found. Run a game first.")
        return

    log_files = list(log_dir.glob("*.json"))
    if not log_files:
        print("No game logs found. Run a game first.")
        return

    # Load most recent
    log_file = max(log_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading: {log_file}")

    reader = GameLogReader.load(str(log_file))

    print(f"\nGame ID: {reader.game_id}")
    print(f"Players: {[p['id'] for p in reader.players]}")
    print(f"Total turns: {reader.total_turns}")

    if reader.result:
        print(f"Winner: {reader.result.get('winner')}")
        print(f"Final scores: {reader.result.get('final_scores')}")

    # Analyze LLM decisions
    decisions = reader.get_llm_decisions()
    if decisions:
        print(f"\nLLM Decisions: {len(decisions)}")
        avg_latency = sum(d.get('latency_ms', 0) for d in decisions) / len(decisions)
        print(f"Avg latency: {avg_latency:.0f}ms")

        # Show first decision's reasoning
        first = decisions[0]
        print(f"\nFirst decision reasoning:")
        print(f"  {first.get('reasoning', 'N/A')[:200]}...")


def main():
    """Run all examples."""
    print("="*60)
    print("CATAN-ARENA QUICKSTART EXAMPLES")
    print("="*60)

    # Check for API keys
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))

    if not has_anthropic and not has_openai:
        print("\nWARNING: No API keys found!")
        print("Set ANTHROPIC_API_KEY or OPENAI_API_KEY to run LLM examples.")
        print("Running random-only example instead.\n")

        # Simple random vs random game
        from catan_arena.orchestration.runner import run_simple_game

        players = [
            {"id": "random_1", "color": "RED", "type": "random"},
            {"id": "random_2", "color": "BLUE", "type": "random"},
        ]

        result = run_simple_game(
            player_configs=players,
            max_turns=100,
            verbose=True,
        )
        print(f"\nResult: {result}")
        return

    # Run examples
    try:
        example_1_simple_game()
    except Exception as e:
        print(f"Example 1 failed: {e}")

    # Only run LLM vs LLM if both keys present
    if has_anthropic and has_openai:
        try:
            example_2_llm_vs_llm()
        except Exception as e:
            print(f"Example 2 failed: {e}")

    try:
        example_5_replay_analysis()
    except Exception as e:
        print(f"Example 5 failed: {e}")


if __name__ == "__main__":
    main()
