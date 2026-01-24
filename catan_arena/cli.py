#!/usr/bin/env python3
"""
Command-line interface for Catan-Arena.

Usage:
    catan-arena run --config config.yaml
    catan-arena run --players "claude:RED,gpt4:BLUE,random:ORANGE,random:WHITE"
    catan-arena replay logs/game_xxx.json
    catan-arena stats logs/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional


def parse_players(players_str: str) -> List[dict]:
    """
    Parse player string like "claude:RED,gpt4:BLUE,random:ORANGE".

    Format: model_or_type:COLOR
    - For LLM: "claude-3-opus:RED" or "gpt4:BLUE"
    - For random: "random:ORANGE"
    """
    players = []
    colors_used = set()

    for i, spec in enumerate(players_str.split(",")):
        spec = spec.strip()
        if ":" not in spec:
            print(f"Invalid player spec '{spec}'. Use format: type:COLOR")
            sys.exit(1)

        model_or_type, color = spec.rsplit(":", 1)
        color = color.upper()

        if color in colors_used:
            print(f"Color {color} used multiple times")
            sys.exit(1)
        colors_used.add(color)

        player_id = f"player_{i}"

        # Determine type
        if model_or_type.lower() in ["random", "r"]:
            players.append({
                "id": player_id,
                "color": color,
                "type": "random",
            })
        elif model_or_type.lower() in ["minimax", "mm"]:
            players.append({
                "id": player_id,
                "color": color,
                "type": "minimax",
            })
        else:
            # Assume LLM
            from catan_arena.llm.providers import resolve_model_alias
            model = resolve_model_alias(model_or_type)
            players.append({
                "id": player_id,
                "color": color,
                "type": "llm",
                "model": model,
            })

    return players


def cmd_run(args):
    """Run a game."""
    from catan_arena.orchestration.runner import run_simple_game, run_arena_game
    from catan_arena.config import load_config

    # Load config from file or parse players
    if args.config:
        config = load_config(args.config)
        result = run_arena_game(config, verbose=not args.quiet)
    elif args.players:
        players = parse_players(args.players)
        result = run_simple_game(
            player_configs=players,
            map_type=args.map_type,
            vps_to_win=args.vps,
            max_turns=args.max_turns,
            seed=args.seed,
            log_dir=args.log_dir,
            verbose=not args.quiet,
        )
    else:
        print("Either --config or --players required")
        sys.exit(1)

    # Output result
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n{'='*50}")
        print(f"Game Complete: {result['game_id']}")
        print(f"Winner: {result['winner']}")
        print(f"Scores: {result['final_scores']}")
        print(f"Turns: {result['total_turns']}")
        if result.get('log_path'):
            print(f"Log: {result['log_path']}")


def cmd_replay(args):
    """Display replay information."""
    from catan_arena.storage.game_log import GameLogReader

    reader = GameLogReader.load(args.log_file)

    print(f"\nGame: {reader.game_id}")
    print(f"Type: {reader.game_type}")
    print(f"Players:")
    for p in reader.players:
        print(f"  - {p['id']} ({p['color']}): {p['type']}")
        if p.get('model'):
            print(f"      Model: {p['model']}")

    print(f"\nTotal turns: {reader.total_turns}")

    if reader.result:
        print(f"\nResult:")
        print(f"  Winner: {reader.result.get('winner')}")
        print(f"  Reason: {reader.result.get('termination_reason')}")
        print(f"  Scores: {reader.result.get('final_scores')}")

        stats = reader.result.get('statistics', {})
        if stats:
            print(f"\nStatistics:")
            print(f"  LLM calls: {stats.get('total_llm_calls', 0)}")
            print(f"  Total tokens: {stats.get('total_tokens_used', 0)}")
            print(f"  Avg decision time: {stats.get('average_decision_time_ms', 0):.0f}ms")

    # Show turns if requested
    if args.turns:
        print(f"\nTurns:")
        for turn in reader.iter_turns():
            action = turn.get('action', {})
            print(f"  {turn['turn_number']}: {turn['player_id']} - {action.get('action_type')}")


def cmd_stats(args):
    """Show statistics for games in a directory."""
    from catan_arena.storage.game_log import GameLogReader

    log_dir = Path(args.log_dir)
    log_files = list(log_dir.glob("*.json"))

    if not log_files:
        print(f"No log files found in {log_dir}")
        return

    print(f"Analyzing {len(log_files)} games...\n")

    wins = {}
    total_turns = 0
    total_tokens = 0
    total_llm_calls = 0

    for log_file in log_files:
        try:
            reader = GameLogReader.load(str(log_file))

            if reader.result:
                winner = reader.result.get('winner')
                if winner:
                    wins[winner] = wins.get(winner, 0) + 1

                total_turns += reader.total_turns
                stats = reader.result.get('statistics', {})
                total_tokens += stats.get('total_tokens_used', 0)
                total_llm_calls += stats.get('total_llm_calls', 0)

        except Exception as e:
            print(f"Error reading {log_file}: {e}")

    print(f"Games analyzed: {len(log_files)}")
    print(f"\nWins:")
    for player, count in sorted(wins.items(), key=lambda x: -x[1]):
        pct = count / len(log_files) * 100
        print(f"  {player}: {count} ({pct:.1f}%)")

    if log_files:
        print(f"\nAverages:")
        print(f"  Turns per game: {total_turns / len(log_files):.1f}")
        print(f"  Tokens per game: {total_tokens / len(log_files):.0f}")
        print(f"  LLM calls per game: {total_llm_calls / len(log_files):.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Catan-Arena: LLM Benchmark for Settlers of Catan"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a game")
    run_parser.add_argument("--config", "-c", help="Config file (YAML or JSON)")
    run_parser.add_argument(
        "--players", "-p",
        help="Players: 'claude:RED,gpt4:BLUE,random:ORANGE,random:WHITE'"
    )
    run_parser.add_argument("--map-type", default="BASE", choices=["BASE", "MINI"])
    run_parser.add_argument("--vps", type=int, default=10, help="VPs to win")
    run_parser.add_argument("--max-turns", type=int, default=500)
    run_parser.add_argument("--seed", type=int, help="Random seed")
    run_parser.add_argument("--log-dir", default="./game_logs")
    run_parser.add_argument("--quiet", "-q", action="store_true")
    run_parser.add_argument("--json", action="store_true", help="Output JSON")

    # Replay command
    replay_parser = subparsers.add_parser("replay", help="View replay info")
    replay_parser.add_argument("log_file", help="Game log file")
    replay_parser.add_argument("--turns", action="store_true", help="Show all turns")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    stats_parser.add_argument("log_dir", help="Directory with log files")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "replay":
        cmd_replay(args)
    elif args.command == "stats":
        cmd_stats(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
