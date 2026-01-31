import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "./LeaderboardScreen.scss";

interface ModelInfo {
  name: string;
  provider: string;
  context_window: number;
  param_count: string;
}

interface GameStat {
  elo: number;
  wins: number;
  losses: number;
  draws: number;
  win_rate: number;
  games: number;
  avg_latency_ms: number;
}

interface LeaderboardEntry {
  rank: number;
  model: ModelInfo;
  elo: number;
  elo_ci: number;
  elo_trend: number;
  wins: number;
  losses: number;
  draws: number;
  win_rate: number;
  avg_latency_ms: number;
  game_stats: Record<string, GameStat>;
}

interface MatchRecord {
  player_1: string;
  player_2: string;
  game_type: string;
  winner: string | null;
  p1_score: number;
  p2_score: number;
  total_turns: number;
  timestamp: string;
}

interface LeaderboardData {
  entries: LeaderboardEntry[];
  matches: MatchRecord[];
  meta: {
    total_matches: number;
    game_types: string[];
    num_models: number;
  };
}

type GameFilter = "all" | "catan" | "codenames" | "simple_card";

const GAME_LABELS: Record<string, string> = {
  all: "All Games",
  catan: "Catan",
  codenames: "Codenames",
  simple_card: "Simple Card",
};

const GAME_ICONS: Record<string, string> = {
  catan: "🏝️",
  codenames: "🕵️",
  simple_card: "🃏",
};

function RankBadge({ rank }: { rank: number }) {
  let cls = "rank-badge";
  if (rank === 1) cls += " gold";
  else if (rank === 2) cls += " silver";
  else if (rank === 3) cls += " bronze";
  return <span className={cls}>#{rank}</span>;
}

function TrendArrow({ value }: { value: number }) {
  if (value > 0) return <span className="trend up">▲ {value}</span>;
  if (value < 0) return <span className="trend down">▼ {value}</span>;
  return <span className="trend neutral">—</span>;
}

export default function LeaderboardScreen() {
  const navigate = useNavigate();
  const [data, setData] = useState<LeaderboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<GameFilter>("all");

  // Unlock scrolling (index.css sets overflow:hidden globally for game screens)
  useEffect(() => {
    document.documentElement.style.overflow = "auto";
    document.body.style.overflow = "auto";
    document.getElementById("root")!.style.overflow = "auto";
    return () => {
      document.documentElement.style.overflow = "";
      document.body.style.overflow = "";
      document.getElementById("root")!.style.overflow = "";
    };
  }, []);

  useEffect(() => {
    fetch("/api/arena/leaderboard")
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((json) => {
        setData(json);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  // Filter entries to only those with games in the selected type
  const filteredEntries = data
    ? filter === "all"
      ? data.entries
      : data.entries.filter((e) => (e.game_stats[filter]?.games ?? 0) > 0)
    : [];

  // Sort by selected game's Elo when filtering, otherwise overall Elo
  const sortedEntries = [...filteredEntries].sort((a, b) => {
    if (filter !== "all") {
      return (b.game_stats[filter]?.elo ?? 0) - (a.game_stats[filter]?.elo ?? 0);
    }
    return b.elo - a.elo;
  });

  const filteredMatches = data
    ? filter === "all"
      ? data.matches
      : data.matches.filter((m) => m.game_type === filter)
    : [];

  if (loading) {
    return (
      <div className="leaderboard-screen loading">
        <div className="spinner"></div>
        <p>Loading leaderboard...</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="leaderboard-screen error-state">
        <p>Failed to load leaderboard: {error}</p>
        <button onClick={() => navigate("/")}>Back</button>
      </div>
    );
  }

  return (
    <div className="leaderboard-screen">
      {/* Header */}
      <header className="lb-header">
        <button className="back-btn" onClick={() => navigate("/")}>
          ← Back
        </button>
        <h1>Leaderboard</h1>
        <p className="lb-meta">
          {data.meta.num_models} models · {data.meta.total_matches} matches ·{" "}
          {data.meta.game_types.length} game types
        </p>
      </header>

      {/* Filter tabs */}
      <div className="filter-tabs">
        {(["all", "catan", "codenames", "simple_card"] as GameFilter[]).map(
          (g) => (
            <button
              key={g}
              className={`filter-tab ${filter === g ? "active" : ""}`}
              onClick={() => setFilter(g)}
            >
              {g !== "all" && <span className="tab-icon">{GAME_ICONS[g]}</span>}
              {GAME_LABELS[g]}
            </button>
          )
        )}
      </div>

      {/* Rankings table */}
      <section className="rankings">
        <div className="rankings-header">
          <span className="col-rank">Rank</span>
          <span className="col-model">Model</span>
          <span className="col-elo">Elo</span>
          <span className="col-trend">Trend</span>
          <span className="col-record">W / L / D</span>
          <span className="col-winrate">Win %</span>
          <span className="col-latency">Latency</span>
        </div>
        {sortedEntries.map((entry, idx) => {
          const displayElo =
            filter !== "all" ? entry.game_stats[filter]?.elo ?? 0 : entry.elo;
          const displayCi =
            filter !== "all" ? "—" : `±${entry.elo_ci}`;
          return (
            <div key={entry.model.name} className="ranking-row">
              <span className="col-rank">
                <RankBadge rank={idx + 1} />
              </span>
              <span className="col-model">
                <div className="model-info">
                  <strong>{entry.model.name}</strong>
                  <span className="provider">{entry.model.provider}</span>
                </div>
              </span>
              <span className="col-elo">
                <span className="elo-value">{displayElo}</span>
                <span className="elo-ci">{displayCi}</span>
              </span>
              <span className="col-trend">
                <TrendArrow value={entry.elo_trend} />
              </span>
              <span className="col-record">
                <span className="wins">{entry.wins}</span>
                <span className="sep">/</span>
                <span className="losses">{entry.losses}</span>
                <span className="sep">/</span>
                <span className="draws">{entry.draws}</span>
              </span>
              <span className="col-winrate">
                {(entry.win_rate * 100).toFixed(1)}%
              </span>
              <span className="col-latency">{entry.avg_latency_ms}ms</span>
            </div>
          );
        })}
      </section>

      {/* Recent matches */}
      <section className="recent-matches">
        <h2>Recent Matches</h2>
        <div className="matches-grid">
          {filteredMatches.map((match, idx) => (
            <div key={idx} className="match-card">
              <span className="match-game-type">
                {GAME_ICONS[match.game_type]} {GAME_LABELS[match.game_type]}
              </span>
              <div className="match-players">
                <div
                  className={`match-player ${
                    match.winner === match.player_1 ? "winner" : ""
                  }`}
                >
                  <span className="player-name">{match.player_1}</span>
                  <span className="player-score">{match.p1_score}</span>
                </div>
                <span className="vs">vs</span>
                <div
                  className={`match-player ${
                    match.winner === match.player_2 ? "winner" : ""
                  }`}
                >
                  <span className="player-name">{match.player_2}</span>
                  <span className="player-score">{match.p2_score}</span>
                </div>
              </div>
              <span className="match-footer">
                {match.winner ? (
                  <span className="match-result">
                    {match.winner} won · {match.total_turns} turns
                  </span>
                ) : (
                  <span className="match-result draw">
                    Draw · {match.total_turns} turns
                  </span>
                )}
              </span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
