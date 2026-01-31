"""
Leaderboard data structures, Elo engine, and mock benchmark generation.

The mock generator simulates a realistic benchmark round:
  1. Each model is assigned a true skill level (per game type).
  2. Match outcomes are drawn probabilistically from a logistic model
     over the skill difference.
  3. Elo ratings are computed by replaying the match history in order.
  4. All per-game stats, head-to-head records, and trends are aggregated.

Everything is seeded; the same seed always produces the same dataset.
"""

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelProfile:
    """Static metadata for an LLM."""

    name: str
    provider: str
    model_id: str
    context_window: int  # tokens
    release_date: str  # YYYY-MM-DD
    param_count: str  # display string, e.g. "~200B"


@dataclass
class GameTypeStats:
    """Per-game-type performance for one model."""

    game_type: str
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    elo: float = 1500.0
    avg_latency_ms: float = 0.0
    avg_turns: float = 0.0

    @property
    def win_rate(self) -> float:
        decisive = self.wins + self.losses
        return self.wins / decisive if decisive > 0 else 0.0


@dataclass
class H2HRecord:
    """Head-to-head record against a single opponent."""

    opponent: str
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def decisive(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        return self.wins / self.decisive if self.decisive > 0 else 0.0


@dataclass
class MatchRecord:
    """A single completed benchmark match."""

    match_id: int
    game_type: str
    player_1: str  # model name
    player_2: str  # model name
    winner: Optional[str]  # None = draw
    p1_score: int
    p2_score: int
    total_turns: int
    p1_avg_latency_ms: float
    p2_avg_latency_ms: float
    timestamp: str  # ISO 8601
    seed: int


@dataclass
class LeaderboardEntry:
    """Full aggregated leaderboard row for one model."""

    rank: int
    model: ModelProfile
    elo: float
    elo_ci: float  # ±95% confidence interval
    wins: int
    losses: int
    draws: int
    games_played: int
    win_rate: float
    avg_latency_ms: float
    elo_trend: float  # Elo change over last 20 games
    game_stats: Dict[str, GameTypeStats]
    h2h: Dict[str, H2HRecord]


# ---------------------------------------------------------------------------
# Elo engine
# ---------------------------------------------------------------------------


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _k_factor(games_played: int) -> int:
    """K=32 for <30 games (volatile), K=16 after (stable)."""
    return 32 if games_played < 30 else 16


def elo_confidence_interval(games_played: int) -> float:
    """Approximate 95% CI width for an Elo rating."""
    return round(350.0 / math.sqrt(max(games_played, 1)), 1)


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------

MODEL_CATALOG: List[ModelProfile] = [
    ModelProfile("Claude 3.5 Sonnet", "Anthropic", "claude-3-5-sonnet-20241022", 200_000, "2024-10-22", "~200B"),
    ModelProfile("GPT-4o", "OpenAI", "gpt-4o-2024-08-06", 128_000, "2024-08-06", "~200B"),
    ModelProfile("DeepSeek V3", "DeepSeek", "deepseek-v3-20241226", 128_000, "2024-12-26", "671B MoE"),
    ModelProfile("Claude 3 Opus", "Anthropic", "claude-3-opus-20240229", 200_000, "2024-02-29", "~300B"),
    ModelProfile("Gemini 1.5 Pro", "Google", "gemini-1.5-pro-002", 1_000_000, "2024-09-24", "~300B"),
    ModelProfile("Qwen 2.5 72B", "Alibaba", "qwen2.5-72b-instruct", 128_000, "2024-09-15", "72B"),
    ModelProfile("Llama 3.1 70B", "Meta", "meta-llama-3.1-70b-instruct", 128_000, "2024-07-18", "70B"),
    ModelProfile("Mistral Large", "Mistral AI", "mistral-large-2407", 128_000, "2024-07-24", "~123B"),
    ModelProfile("GPT-4o-mini", "OpenAI", "gpt-4o-mini-2024-07-18", 128_000, "2024-07-18", "~8B"),
    ModelProfile("Gemini 1.5 Flash", "Google", "gemini-1.5-flash-002", 1_000_000, "2024-09-24", "~30B"),
    ModelProfile("Llama 3.1 8B", "Meta", "meta-llama-3.1-8b-instruct", 128_000, "2024-07-18", "8B"),
    ModelProfile("Mistral 7B", "Mistral AI", "mistral-7b-v0.3", 32_000, "2024-05-15", "7B"),
]

GAME_TYPES = ["catan", "codenames", "simple_card"]

# ---------------------------------------------------------------------------
# Simulation parameters (mock data only)
# ---------------------------------------------------------------------------

# (base_skill, base_latency_ms)
# base_skill: higher = stronger; fed into a logistic to produce win %
# base_latency_ms: median per-turn response time in the simulation
_SIM_PARAMS: Dict[str, Tuple[float, int]] = {
    "Claude 3.5 Sonnet": (0.72, 850),
    "GPT-4o": (0.68, 620),
    "DeepSeek V3": (0.65, 1100),
    "Claude 3 Opus": (0.63, 1800),
    "Gemini 1.5 Pro": (0.60, 950),
    "Qwen 2.5 72B": (0.57, 1050),
    "Llama 3.1 70B": (0.55, 980),
    "Mistral Large": (0.52, 870),
    "GPT-4o-mini": (0.44, 320),
    "Gemini 1.5 Flash": (0.42, 280),
    "Llama 3.1 8B": (0.33, 400),
    "Mistral 7B": (0.30, 350),
}

# Additive per-game adjustments — models have different relative strengths
_GAME_ADJUSTMENTS: Dict[str, Dict[str, float]] = {
    "catan": {
        "Claude 3.5 Sonnet": 0.04,
        "GPT-4o": 0.02,
        "DeepSeek V3": 0.06,
        "Claude 3 Opus": 0.05,
        "Qwen 2.5 72B": -0.02,
        "Llama 3.1 70B": -0.03,
    },
    "codenames": {
        "Claude 3.5 Sonnet": 0.02,
        "GPT-4o": 0.04,
        "Gemini 1.5 Pro": 0.03,
        "Mistral Large": 0.02,
        "DeepSeek V3": -0.02,
        "Claude 3 Opus": 0.01,
    },
    "simple_card": {
        "GPT-4o": 0.03,
        "Llama 3.1 70B": 0.02,
        "Gemini 1.5 Flash": 0.01,
        "Claude 3.5 Sonnet": -0.01,
        "DeepSeek V3": 0.01,
    },
}

# Base draw probability per game type (scaled up when skills are close)
_DRAW_RATES: Dict[str, float] = {
    "catan": 0.02,
    "codenames": 0.03,
    "simple_card": 0.18,
}

# Total-turn ranges per game type
_TURN_RANGES: Dict[str, Tuple[int, int]] = {
    "catan": (40, 200),
    "codenames": (12, 55),
    "simple_card": (6, 6),  # 3 rounds × 2 half-turns
}

# Winner score range per game type
_WIN_SCORES: Dict[str, Tuple[int, int]] = {
    "catan": (10, 13),  # Victory Points
    "codenames": (7, 9),  # Agents found
    "simple_card": (2, 3),  # Rounds won
}

# Draw score range (both players receive same value)
_DRAW_SCORES: Dict[str, Tuple[int, int]] = {
    "catan": (7, 9),  # Near but under 10 VP (turn-limit draw)
    "codenames": (4, 7),  # Equal agents found
    "simple_card": (1, 1),  # 1-1 after three rounds
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-min(max(x, -20.0), 20.0)))


def _effective_skill(model: str, game_type: str) -> float:
    base, _ = _SIM_PARAMS[model]
    return base + _GAME_ADJUSTMENTS.get(game_type, {}).get(model, 0.0)


# ---------------------------------------------------------------------------
# Benchmark generator
# ---------------------------------------------------------------------------


def generate_benchmark(
    num_matches: int = 600,
    seed: int = 20260131,
) -> Tuple[List[MatchRecord], List[LeaderboardEntry]]:
    """
    Simulate a full benchmark and compute the leaderboard.

    Args:
        num_matches: Total matches to simulate
        seed: Random seed (deterministic)

    Returns:
        (match_records sorted by timestamp, leaderboard_entries sorted by Elo)
    """
    rng = random.Random(seed)
    names = [m.name for m in MODEL_CATALOG]
    catalog = {m.name: m for m in MODEL_CATALOG}

    end_dt = datetime(2026, 1, 31, 18, 0, 0)
    start_dt = end_dt - timedelta(days=30)

    # ------------------------------------------------------------------
    # Phase 1: simulate matches
    # ------------------------------------------------------------------
    matches: List[MatchRecord] = []

    for i in range(num_matches):
        p1, p2 = rng.sample(names, 2)
        gt = rng.choice(GAME_TYPES)

        # Win probability via logistic over skill difference
        s1 = _effective_skill(p1, gt)
        s2 = _effective_skill(p2, gt)
        p1_win_prob = _sigmoid((s1 - s2) * 8)

        # Draw probability (higher when skills are close)
        closeness = 1.0 - min(abs(s1 - s2) * 10, 1.0)
        draw_prob = min(_DRAW_RATES[gt] * (1.0 + 0.5 * closeness), 0.28)

        # Resolve outcome
        p1_dec = p1_win_prob * (1 - draw_prob)
        p2_dec = (1 - p1_win_prob) * (1 - draw_prob)
        roll = rng.random()

        w_min, w_max = _WIN_SCORES[gt]
        if roll < p1_dec:
            winner = p1
            p1_sc = rng.randint(w_min, w_max)
            p2_sc = rng.randint(0, max(p1_sc - 1, 0))
        elif roll < p1_dec + p2_dec:
            winner = p2
            p2_sc = rng.randint(w_min, w_max)
            p1_sc = rng.randint(0, max(p2_sc - 1, 0))
        else:
            winner = None
            d_min, d_max = _DRAW_SCORES[gt]
            sc = rng.randint(d_min, d_max)
            p1_sc = p2_sc = sc

        # Latency: base × uniform jitter
        _, lat1 = _SIM_PARAMS[p1]
        _, lat2 = _SIM_PARAMS[p2]

        # Turns
        t_min, t_max = _TURN_RANGES[gt]
        turns = rng.randint(t_min, t_max)

        # Timestamp spread evenly across 30 days with ±30 min jitter
        frac = i / num_matches
        ts = start_dt + (end_dt - start_dt) * frac
        ts += timedelta(seconds=rng.randint(-1800, 1800))

        matches.append(MatchRecord(
            match_id=i + 1,
            game_type=gt,
            player_1=p1,
            player_2=p2,
            winner=winner,
            p1_score=p1_sc,
            p2_score=p2_sc,
            total_turns=turns,
            p1_avg_latency_ms=round(lat1 * rng.uniform(0.7, 1.4), 1),
            p2_avg_latency_ms=round(lat2 * rng.uniform(0.7, 1.4), 1),
            timestamp=ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            seed=rng.randint(1, 999_999),
        ))

    matches.sort(key=lambda m: m.timestamp)

    # ------------------------------------------------------------------
    # Phase 2: replay history to compute Elo and aggregate stats
    # ------------------------------------------------------------------
    elo = {n: 1500.0 for n in names}
    gp = {n: 0 for n in names}
    elo_hist: Dict[str, List[float]] = {n: [1500.0] for n in names}

    gt_elo = {gt: {n: 1500.0 for n in names} for gt in GAME_TYPES}
    gt_gp = {gt: {n: 0 for n in names} for gt in GAME_TYPES}

    w = {n: 0 for n in names}
    l = {n: 0 for n in names}
    d = {n: 0 for n in names}
    gt_w = {gt: {n: 0 for n in names} for gt in GAME_TYPES}
    gt_l = {gt: {n: 0 for n in names} for gt in GAME_TYPES}
    gt_d = {gt: {n: 0 for n in names} for gt in GAME_TYPES}

    lat_sum = {n: 0.0 for n in names}
    lat_cnt = {n: 0 for n in names}
    gt_lat_sum = {gt: {n: 0.0 for n in names} for gt in GAME_TYPES}
    gt_lat_cnt = {gt: {n: 0 for n in names} for gt in GAME_TYPES}
    gt_turn_sum = {gt: {n: 0 for n in names} for gt in GAME_TYPES}

    h2h: Dict[str, Dict[str, H2HRecord]] = {n: {} for n in names}

    for m in matches:
        p1, p2, gt = m.player_1, m.player_2, m.game_type

        # Scores for Elo
        if m.winner == p1:
            s1, s2 = 1.0, 0.0
            w[p1] += 1; l[p2] += 1
            gt_w[gt][p1] += 1; gt_l[gt][p2] += 1
        elif m.winner == p2:
            s1, s2 = 0.0, 1.0
            w[p2] += 1; l[p1] += 1
            gt_w[gt][p2] += 1; gt_l[gt][p1] += 1
        else:
            s1, s2 = 0.5, 0.5
            d[p1] += 1; d[p2] += 1
            gt_d[gt][p1] += 1; gt_d[gt][p2] += 1

        # Overall Elo — read old ratings, then write both updates
        r1, r2 = elo[p1], elo[p2]
        e1 = _expected_score(r1, r2)
        e2 = _expected_score(r2, r1)
        elo[p1] = r1 + _k_factor(gp[p1]) * (s1 - e1)
        elo[p2] = r2 + _k_factor(gp[p2]) * (s2 - e2)
        gp[p1] += 1; gp[p2] += 1
        elo_hist[p1].append(elo[p1])
        elo_hist[p2].append(elo[p2])

        # Per-game-type Elo
        gr1, gr2 = gt_elo[gt][p1], gt_elo[gt][p2]
        ge1 = _expected_score(gr1, gr2)
        ge2 = _expected_score(gr2, gr1)
        gt_elo[gt][p1] = gr1 + _k_factor(gt_gp[gt][p1]) * (s1 - ge1)
        gt_elo[gt][p2] = gr2 + _k_factor(gt_gp[gt][p2]) * (s2 - ge2)
        gt_gp[gt][p1] += 1; gt_gp[gt][p2] += 1

        # Latency
        lat_sum[p1] += m.p1_avg_latency_ms; lat_cnt[p1] += 1
        lat_sum[p2] += m.p2_avg_latency_ms; lat_cnt[p2] += 1
        gt_lat_sum[gt][p1] += m.p1_avg_latency_ms; gt_lat_cnt[gt][p1] += 1
        gt_lat_sum[gt][p2] += m.p2_avg_latency_ms; gt_lat_cnt[gt][p2] += 1

        # Turns
        gt_turn_sum[gt][p1] += m.total_turns
        gt_turn_sum[gt][p2] += m.total_turns

        # Head-to-head
        if p2 not in h2h[p1]:
            h2h[p1][p2] = H2HRecord(opponent=p2)
        if p1 not in h2h[p2]:
            h2h[p2][p1] = H2HRecord(opponent=p1)

        if m.winner == p1:
            h2h[p1][p2].wins += 1; h2h[p2][p1].losses += 1
        elif m.winner == p2:
            h2h[p1][p2].losses += 1; h2h[p2][p1].wins += 1
        else:
            h2h[p1][p2].draws += 1; h2h[p2][p1].draws += 1

    # ------------------------------------------------------------------
    # Phase 3: build LeaderboardEntry objects
    # ------------------------------------------------------------------
    entries: List[LeaderboardEntry] = []

    for name in names:
        total = gp[name]
        decisive = w[name] + l[name]

        # Trend: Elo change over last 20 games played
        hist = elo_hist[name]
        lookback = min(20, len(hist) - 1)
        trend = round(hist[-1] - hist[-(lookback + 1)], 1) if lookback > 0 else 0.0

        # Per-game stats
        gs: Dict[str, GameTypeStats] = {}
        for gt in GAME_TYPES:
            cnt = gt_gp[gt][name]
            gs[gt] = GameTypeStats(
                game_type=gt,
                games=cnt,
                wins=gt_w[gt][name],
                losses=gt_l[gt][name],
                draws=gt_d[gt][name],
                elo=round(gt_elo[gt][name], 1),
                avg_latency_ms=round(
                    gt_lat_sum[gt][name] / max(gt_lat_cnt[gt][name], 1), 1
                ),
                avg_turns=round(
                    gt_turn_sum[gt][name] / max(gt_gp[gt][name], 1), 1
                ),
            )

        entries.append(LeaderboardEntry(
            rank=0,  # assigned after sort
            model=catalog[name],
            elo=round(elo[name], 1),
            elo_ci=elo_confidence_interval(total),
            wins=w[name],
            losses=l[name],
            draws=d[name],
            games_played=total,
            win_rate=w[name] / decisive if decisive > 0 else 0.0,
            avg_latency_ms=round(lat_sum[name] / max(lat_cnt[name], 1), 1),
            elo_trend=trend,
            game_stats=gs,
            h2h=h2h[name],
        ))

    entries.sort(key=lambda e: -e.elo)
    for i, e in enumerate(entries):
        e.rank = i + 1

    return matches, entries
