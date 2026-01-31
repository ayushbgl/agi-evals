import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "./HomePage.scss";

interface GameDef {
  id: string;
  title: string;
  description: string;
  tags: string[];
  gradient: string;
  icon: string;
  playRoute: string | null;
  replayRoute: string | null;
}

const GAMES: GameDef[] = [
  {
    id: "catan",
    title: "Settlers of Catan",
    description:
      "Collect resources, build settlements, and trade your way to victory in this classic strategy board game.",
    tags: ["Strategy", "Trading", "2–4 Players"],
    gradient: "linear-gradient(145deg, #0d3b23 0%, #1a6b3c 50%, #2a8a4e 100%)",
    icon: "🏝️",
    playRoute: "/catan/play",
    replayRoute: "/catan/replay",
  },
  {
    id: "codenames",
    title: "Codenames",
    description:
      "Spymasters give one-word clues to guide their operatives to the team's secret agents on the board.",
    tags: ["Word Game", "Team Play", "Deduction"],
    gradient: "linear-gradient(145deg, #1a1040 0%, #2d1b69 50%, #3d2b8a 100%)",
    icon: "🕵️",
    playRoute: "/codenames/play",
    replayRoute: "/codenames/replay",
  },
  {
    id: "simple_card",
    title: "Simple Card",
    description:
      "A fast-paced number battle. Play the highest card each round to outscore your opponent.",
    tags: ["Card Game", "2 Players", "Quick"],
    gradient: "linear-gradient(145deg, #4a0e1e 0%, #8b1a35 50%, #c62a47 100%)",
    icon: "🃏",
    playRoute: null,
    replayRoute: null,
  },
];

function GameCard({
  game,
  variant,
}: {
  game: GameDef;
  variant: "play" | "replay";
}) {
  const navigate = useNavigate();
  const route = variant === "play" ? game.playRoute : game.replayRoute;
  const disabled = !route;

  return (
    <div
      className={`game-card ${disabled ? "disabled" : ""}`}
      style={{ background: game.gradient }}
      onClick={() => route && navigate(route)}
    >
      <span className="card-icon">{game.icon}</span>

      {disabled && <span className="coming-soon">Coming Soon</span>}

      <div className="card-body">
        <h3>{game.title}</h3>
        <p>{game.description}</p>
        <div className="card-tags">
          {game.tags.map((t) => (
            <span key={t} className="tag">
              {t}
            </span>
          ))}
        </div>
      </div>

      {!disabled && (
        <div className="card-overlay">
          <span className="play-btn">
            {variant === "play" ? "▶  Play" : "▶  Replay"}
          </span>
        </div>
      )}
    </div>
  );
}

export default function HomePage() {
  const navigate = useNavigate();
  const hero = GAMES[0];

  // index.css sets overflow:hidden on html/body/#root for game screens;
  // unlock it while the scrollable homepage is mounted.
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

  return (
    <div className="home-page">
      {/* Header */}
      <header className="hp-header">
        <h1 className="logo">AGI-Evals</h1>
        <nav>
          <span>Games</span>
          <span>Replays</span>
        </nav>
      </header>

      {/* Hero banner */}
      <section className="hero" style={{ background: hero.gradient }}>
        <div className="hero-body">
          <span className="hero-badge">FEATURED</span>
          <h2>{hero.title}</h2>
          <p>{hero.description}</p>
          <div className="hero-tags">
            {hero.tags.map((t) => (
              <span key={t} className="tag">
                {t}
              </span>
            ))}
          </div>
          <div className="hero-actions">
            <button
              className="btn-play"
              onClick={() => navigate(hero.playRoute!)}
            >
              ▶  Play Now
            </button>
            <button
              className="btn-secondary"
              onClick={() => navigate(hero.replayRoute!)}
            >
              Replay
            </button>
          </div>
        </div>
        <span className="hero-icon">{hero.icon}</span>
      </section>

      {/* Play row */}
      <section className="row">
        <h3 className="row-title">Play Games</h3>
        <div className="row-cards">
          {GAMES.map((g) => (
            <GameCard key={g.id} game={g} variant="play" />
          ))}
        </div>
      </section>

      {/* Replay row */}
      <section className="row">
        <h3 className="row-title">Replays</h3>
        <div className="row-cards">
          {GAMES.map((g) => (
            <GameCard key={g.id} game={g} variant="replay" />
          ))}
        </div>
      </section>

      {/* Leaderboard row */}
      <section className="row">
        <h3 className="row-title">Leaderboard</h3>
        <div className="row-cards">
          <div className="lb-card" onClick={() => navigate("/leaderboard")}>
            <span className="lb-icon">🏆</span>
            <div className="lb-body">
              <h3>AGI-Evals Leaderboard</h3>
              <p>
                Elo rankings across 12 models · 3 game types · 600+ matches
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
