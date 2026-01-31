"""
Rich terminal display for the AGI-Evals leaderboard.

Renders four sections:
  1. Overall rankings table (Elo, W/L/D, win%, latency, trend)
  2. Per-game-type Elo breakdown
  3. Head-to-head rivalries (top 10 by total games played)
  4. Most recent matches
"""

from typing import List, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from arena.leaderboard.data import LeaderboardEntry, MatchRecord, GAME_TYPES


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _elo_color(elo: float) -> str:
    if elo >= 1700:
        return "bold bright_green"
    if elo >= 1600:
        return "green"
    if elo >= 1500:
        return "yellow"
    if elo >= 1400:
        return "orange1"
    return "red"


def _trend_markup(trend: float) -> str:
    if trend >= 10:
        return f"[bold bright_green]+{trend:.0f} ▲[/]"
    if trend >= 3:
        return f"[green]+{trend:.0f} ▲[/]"
    if trend > -3:
        return "[dim]—[/]"
    if trend > -10:
        return f"[orange1]{trend:.0f} ▼[/]"
    return f"[bold red]{trend:.0f} ▼[/]"


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _rank_badge(rank: int) -> str:
    if rank == 1:
        return "[bold gold1]#1[/]"
    if rank == 2:
        return "[bold bright_white]#2[/]"
    if rank == 3:
        return "[bold orange1]#3[/]"
    return f"[dim]#{rank}[/]"


def _game_label(gt: str) -> str:
    return gt.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------


def render_leaderboard(
    entries: List[LeaderboardEntry],
    matches: List[MatchRecord],
    console: Optional[Console] = None,
) -> None:
    """Render the complete leaderboard to the terminal."""
    if console is None:
        console = Console()

    total_draws = sum(e.draws for e in entries) // 2  # each draw counted twice

    # === Header panel =====================================================
    console.print()
    console.print(Panel(
        Text.assemble(
            ("AGI-EVALS BENCHMARK LEADERBOARD", "bold bright_white"),
            "\n\n",
            (f"{len(matches)} matches played", "cyan"),
            ("  •  ", "dim"),
            (f"{len(GAME_TYPES)} game types", "cyan"),
            ("  •  ", "dim"),
            (f"{len(entries)} models", "cyan"),
            ("  •  ", "dim"),
            ("Jan 2 – Jan 31, 2026", "cyan"),
            "\n",
            ("Elo rating system  •  K=32 (<30 games), K=16 (30+)  •  95% CI shown", "dim"),
        ),
        title="[bold bright_cyan]  AGI-EVALS  [/]",
        subtitle=f"[dim]{len(matches) - total_draws} decisive  •  {total_draws} draws[/]",
        border_style="cyan",
        expand=False,
        padding=(0, 2),
    ))
    console.print()

    # === Overall rankings =================================================
    main = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold dim",
        title="[bold]Overall Rankings[/]",
        min_width=105,
        pad_edge=True,
    )
    main.add_column("#", width=4, justify="right")
    main.add_column("Model", width=25)
    main.add_column("Provider", width=13, style="dim")
    main.add_column("Elo", width=7, justify="right")
    main.add_column("±CI", width=6, justify="right", style="dim")
    main.add_column("W", width=5, justify="right", style="green")
    main.add_column("L", width=5, justify="right", style="red")
    main.add_column("D", width=4, justify="right", style="dim")
    main.add_column("Win%", width=7, justify="right")
    main.add_column("Avg Lat", width=9, justify="right", style="dim")
    main.add_column("Trend", width=10, justify="right")

    for e in entries:
        color = _elo_color(e.elo)
        main.add_row(
            _rank_badge(e.rank),
            e.model.name,
            e.model.provider,
            f"[{color}]{e.elo:.0f}[/]",
            f"±{e.elo_ci:.0f}",
            str(e.wins),
            str(e.losses),
            str(e.draws),
            _pct(e.win_rate),
            f"{e.avg_latency_ms:.0f} ms",
            _trend_markup(e.elo_trend),
        )

    console.print(main)
    console.print()

    # === Per-game-type Elo ================================================
    gt_table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold dim",
        title="[bold]Elo by Game Type[/]",
        min_width=105,
    )
    gt_table.add_column("Model", width=25, style="bold")
    for gt in GAME_TYPES:
        gt_table.add_column(_game_label(gt), width=26, justify="center")

    for e in entries:
        row = [e.model.name]
        for gt in GAME_TYPES:
            s = e.game_stats.get(gt)
            if s and s.games > 0:
                c = _elo_color(s.elo)
                row.append(
                    f"[{c}]{s.elo:.0f}[/] "
                    f"[dim]({_pct(s.win_rate)} {s.games}g {s.avg_latency_ms:.0f}ms)[/]"
                )
            else:
                row.append("[dim]—[/]")
        gt_table.add_row(*row)

    console.print(gt_table)
    console.print()

    # === Head-to-head rivalries ============================================
    rivalries: list = []
    seen: set = set()
    # entries is sorted by rank — first time we encounter a pair the current
    # entry is the higher-ranked model, so its h2h record is the one we keep.
    for e in entries:
        for opp, rec in e.h2h.items():
            pair = tuple(sorted([e.model.name, opp]))
            if pair in seen:
                continue
            seen.add(pair)
            rivalries.append((e.model.name, opp, rec.wins, rec.losses, rec.draws))

    rivalries.sort(key=lambda r: -(r[2] + r[3] + r[4]))
    rivalries = rivalries[:10]

    rank_of = {e.model.name: e.rank for e in entries}

    h2h_table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold dim",
        title="[bold]Head-to-Head Rivalries[/] (top 10 by games played)",
        min_width=105,
    )
    h2h_table.add_column("Matchup", width=56, no_wrap=True)
    h2h_table.add_column("Record", width=12, justify="center")
    h2h_table.add_column("Win%", width=8, justify="center")
    h2h_table.add_column("Games", width=7, justify="right", style="dim")

    for left, right, lw, rw, draws in rivalries:
        decisive = lw + rw
        wr = lw / decisive if decisive > 0 else 0.5
        total = lw + rw + draws
        wr_color = "green" if wr >= 0.6 else ("red" if wr <= 0.4 else "yellow")

        matchup = Text()
        matchup.append(f"#{rank_of[left]} ", style="dim")
        matchup.append(left, style="bold")
        matchup.append("  vs  ", style="dim")
        matchup.append(f"#{rank_of[right]} ", style="dim")
        matchup.append(right, style="bold")

        record = f"{lw}-{rw}" + (f"-{draws}" if draws else "")

        h2h_table.add_row(
            matchup,
            record,
            f"[{wr_color}]{_pct(wr)}[/]",
            str(total),
        )

    console.print(h2h_table)
    console.print()

    # === Recent matches ====================================================
    recent = matches[-10:]

    rec_table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold dim",
        title="[bold]Recent Matches[/]",
        min_width=105,
    )
    rec_table.add_column("Game", width=14)
    rec_table.add_column("Player 1", width=25)
    rec_table.add_column("Player 2", width=25)
    rec_table.add_column("Score", width=9, justify="center")
    rec_table.add_column("Winner", width=22)
    rec_table.add_column("Turns", width=6, justify="right", style="dim")

    for m in reversed(recent):
        if m.winner == m.player_1:
            p1s, p2s = "bold green", "dim"
            result = f"[green]{m.winner}[/]"
        elif m.winner == m.player_2:
            p1s, p2s = "dim", "bold green"
            result = f"[green]{m.winner}[/]"
        else:
            p1s, p2s = "", ""
            result = "[dim italic]Draw[/]"

        rec_table.add_row(
            _game_label(m.game_type),
            f"[{p1s}]{m.player_1}[/]",
            f"[{p2s}]{m.player_2}[/]",
            f"{m.p1_score}\u2013{m.p2_score}",
            result,
            str(m.total_turns),
        )

    console.print(rec_table)
    console.print()

    # === Footer ============================================================
    top = entries[0]
    console.print(
        f"[dim]Top model: [bold]{top.model.name}[/bold] ({top.elo:.0f} Elo)  •  "
        f"Provider: {top.model.provider}  •  "
        f"Context: {top.model.context_window:,} tokens  •  "
        f"Params: {top.model.param_count}[/]"
    )
    console.print()
