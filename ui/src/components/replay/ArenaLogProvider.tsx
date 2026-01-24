/**
 * Arena Log Provider - Context for managing game log state
 *
 * Provides game log data, playback controls, and view mode management
 * to child components for the Arena replay viewer.
 */

import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useMemo,
  useCallback,
  ReactNode,
} from "react";

import type {
  ArenaGameLog,
  TurnRecord,
  ViewMode,
  VisibleGameState,
  ArenaPlayer,
} from "../../utils/arenaApi.types";

interface ArenaLogContextValue {
  // Data
  gameLog: ArenaGameLog | null;
  loading: boolean;
  error: string | null;
  currentTurn: number;
  totalTurns: number;
  currentTurnData: TurnRecord | null;

  // Playback controls
  isPlaying: boolean;
  playbackSpeed: number;
  play: () => void;
  pause: () => void;
  seekToTurn: (turn: number) => void;
  nextTurn: () => void;
  prevTurn: () => void;
  setPlaybackSpeed: (speed: number) => void;

  // View mode (Fog of War)
  viewMode: ViewMode;
  setViewMode: (mode: ViewMode) => void;

  // Computed state based on view mode
  visibleState: VisibleGameState | null;
}

const ArenaLogContext = createContext<ArenaLogContextValue | null>(null);

export function useArenaLog(): ArenaLogContextValue {
  const context = useContext(ArenaLogContext);
  if (!context) {
    throw new Error("useArenaLog must be used within ArenaLogProvider");
  }
  return context;
}

interface ArenaLogProviderProps {
  children: ReactNode;
  logData?: ArenaGameLog; // Direct data
  logUrl?: string; // URL to fetch
}

export function ArenaLogProvider({
  children,
  logData,
  logUrl,
}: ArenaLogProviderProps) {
  const [gameLog, setGameLog] = useState<ArenaGameLog | null>(logData || null);
  const [loading, setLoading] = useState(!logData && !!logUrl);
  const [error, setError] = useState<string | null>(null);
  const [currentTurn, setCurrentTurn] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [viewMode, setViewMode] = useState<ViewMode>({ type: "omniscient" });

  // Load game log from URL
  useEffect(() => {
    if (logData) {
      setGameLog(logData);
      setLoading(false);
      return;
    }

    if (!logUrl) return;

    setLoading(true);
    setError(null);

    fetch(logUrl)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: ArenaGameLog) => {
        setGameLog(data);
        setCurrentTurn(0);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, [logUrl, logData]);

  // Playback timer
  useEffect(() => {
    if (!isPlaying || !gameLog) return;

    const interval = setInterval(() => {
      setCurrentTurn((prev) => {
        if (prev >= gameLog.turns.length - 1) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 1000 / playbackSpeed);

    return () => clearInterval(interval);
  }, [isPlaying, playbackSpeed, gameLog]);

  // Current turn data
  const currentTurnData = useMemo(() => {
    if (!gameLog || currentTurn >= gameLog.turns.length) {
      return null;
    }
    return gameLog.turns[currentTurn];
  }, [gameLog, currentTurn]);

  // Compute visible state based on view mode
  const visibleState = useMemo(() => {
    if (!gameLog || !currentTurnData) {
      return null;
    }
    return computeVisibleState(
      currentTurnData,
      viewMode,
      gameLog.players
    );
  }, [gameLog, currentTurnData, viewMode]);

  // Playback controls
  const play = useCallback(() => setIsPlaying(true), []);
  const pause = useCallback(() => setIsPlaying(false), []);

  const seekToTurn = useCallback(
    (turn: number) => {
      if (!gameLog) return;
      setCurrentTurn(Math.max(0, Math.min(turn, gameLog.turns.length - 1)));
    },
    [gameLog]
  );

  const nextTurn = useCallback(() => {
    if (!gameLog) return;
    setCurrentTurn((t) => Math.min(t + 1, gameLog.turns.length - 1));
  }, [gameLog]);

  const prevTurn = useCallback(() => {
    setCurrentTurn((t) => Math.max(t - 1, 0));
  }, []);

  const value: ArenaLogContextValue = {
    gameLog,
    loading,
    error,
    currentTurn,
    totalTurns: gameLog?.turns.length ?? 0,
    currentTurnData,
    isPlaying,
    playbackSpeed,
    play,
    pause,
    seekToTurn,
    nextTurn,
    prevTurn,
    setPlaybackSpeed,
    viewMode,
    setViewMode,
    visibleState,
  };

  return (
    <ArenaLogContext.Provider value={value}>
      {children}
    </ArenaLogContext.Provider>
  );
}

/**
 * Compute what's visible based on view mode (fog of war)
 */
function computeVisibleState(
  turnData: TurnRecord,
  viewMode: ViewMode,
  players: ArenaPlayer[]
): VisibleGameState {
  const { public_state, private_states } = turnData;

  switch (viewMode.type) {
    case "omniscient":
      // Show everything - all players' hands visible
      return {
        board: public_state.board,
        playerStates: Object.fromEntries(
          players.map((p) => [
            p.id,
            {
              ...public_state.player_summaries[p.id],
              ...private_states[p.id],
              handVisible: true,
            },
          ])
        ),
        showAllReasoning: true,
      };

    case "player_pov":
      // Show only what the selected player could see
      const povPlayerId = viewMode.player_id || players[0]?.id;
      return {
        board: public_state.board,
        playerStates: Object.fromEntries(
          players.map((p) => {
            if (p.id === povPlayerId) {
              return [
                p.id,
                {
                  ...public_state.player_summaries[p.id],
                  ...private_states[p.id],
                  handVisible: true,
                },
              ];
            } else {
              return [
                p.id,
                {
                  ...public_state.player_summaries[p.id],
                  handVisible: false,
                },
              ];
            }
          })
        ),
        showAllReasoning: false,
        showReasoningFor: povPlayerId,
      };

    case "spectator":
      // Show only public information
      return {
        board: public_state.board,
        playerStates: Object.fromEntries(
          players.map((p) => [
            p.id,
            {
              ...public_state.player_summaries[p.id],
              handVisible: false,
            },
          ])
        ),
        showAllReasoning: false,
      };

    default:
      return {
        board: public_state.board,
        playerStates: {},
        showAllReasoning: false,
      };
  }
}

export default ArenaLogProvider;
