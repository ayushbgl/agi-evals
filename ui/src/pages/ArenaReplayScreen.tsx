/**
 * Arena Replay Screen - Main page for viewing Arena game replays
 *
 * Displays Catan-Arena game logs with:
 * - Board visualization (reusing existing ZoomableBoard)
 * - View mode controller (Fog of War)
 * - LLM reasoning panel
 * - Playback controls
 * - Turn timeline
 */

import React, { useState, useEffect, useCallback } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { GridLoader } from "react-spinners";
import {
  Box,
  Container,
  Paper,
  Typography,
  Alert,
  Tabs,
  Tab,
  Divider,
} from "@mui/material";

import { ArenaLogProvider, useArenaLog } from "../components/replay/ArenaLogProvider";
import ViewModeController from "../components/replay/ViewModeController";
import LLMReasoningPanel from "../components/replay/LLMReasoningPanel";
import PlaybackControls from "../components/replay/PlaybackControls";
import TurnTimeline from "../components/replay/TurnTimeline";

import type { ArenaGameLog } from "../utils/arenaApi.types";

// Inner component that uses the ArenaLog context
function ArenaReplayContent() {
  const {
    gameLog,
    loading,
    error,
    currentTurn,
    totalTurns,
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
  } = useArenaLog();

  const [activeTab, setActiveTab] = useState(0);

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if user is typing in an input
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      switch (e.key) {
        case " ":
        case "k":
          e.preventDefault();
          isPlaying ? pause() : play();
          break;
        case "ArrowLeft":
        case "j":
          e.preventDefault();
          prevTurn();
          break;
        case "ArrowRight":
        case "l":
          e.preventDefault();
          nextTurn();
          break;
        case "Home":
          e.preventDefault();
          seekToTurn(0);
          break;
        case "End":
          e.preventDefault();
          seekToTurn(totalTurns - 1);
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isPlaying, play, pause, prevTurn, nextTurn, seekToTurn, totalTurns]);

  if (loading) {
    return (
      <Box
        sx={{
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          minHeight: "60vh",
        }}
      >
        <GridLoader color="#1976d2" size={50} />
      </Box>
    );
  }

  if (error) {
    return (
      <Container maxWidth="md" sx={{ mt: 4 }}>
        <Alert severity="error">
          Failed to load game log: {error}
        </Alert>
      </Container>
    );
  }

  if (!gameLog) {
    return (
      <Container maxWidth="md" sx={{ mt: 4 }}>
        <Alert severity="info">
          No game log loaded. Provide a log URL or data.
        </Alert>
      </Container>
    );
  }

  // Get current player info
  const currentPlayer = currentTurnData
    ? gameLog.players.find((p) => p.id === currentTurnData.player_id)
    : null;

  // Determine if we should show reasoning based on view mode
  const shouldShowReasoning =
    viewMode.type === "omniscient" ||
    (viewMode.type === "player_pov" &&
      currentTurnData?.player_id === viewMode.player_id);

  return (
    <Container maxWidth="xl" sx={{ py: 2 }}>
      {/* Header */}
      <Paper elevation={1} sx={{ p: 2, mb: 2 }}>
        <Box
          sx={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: 2,
          }}
        >
          <Box>
            <Typography variant="h5" component="h1">
              Arena Replay: {gameLog.game_id}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {gameLog.game_type} | {gameLog.players.length} players |{" "}
              {gameLog.turns.length} turns
            </Typography>
          </Box>

          <ViewModeController
            viewMode={viewMode}
            onViewModeChange={setViewMode}
            players={gameLog.players}
          />
        </Box>
      </Paper>

      {/* Main content area */}
      <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap" }}>
        {/* Left: Board visualization placeholder */}
        <Paper
          elevation={2}
          sx={{
            flex: "1 1 600px",
            minHeight: 400,
            p: 2,
            display: "flex",
            flexDirection: "column",
          }}
        >
          <Typography variant="subtitle1" sx={{ mb: 1, fontWeight: "bold" }}>
            Game Board
          </Typography>

          {/* Board state summary */}
          {visibleState && (
            <Box sx={{ flex: 1 }}>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Turn {currentTurn + 1} | Phase: {currentTurnData?.phase || "unknown"}
              </Typography>

              {/* Player states */}
              <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
                {gameLog.players.map((player) => {
                  const pState = visibleState.playerStates[player.id];
                  if (!pState) return null;

                  return (
                    <Paper
                      key={player.id}
                      variant="outlined"
                      sx={{
                        p: 1.5,
                        borderLeft: `4px solid ${
                          player.color === "RED"
                            ? "#e53935"
                            : player.color === "BLUE"
                            ? "#1e88e5"
                            : player.color === "ORANGE"
                            ? "#fb8c00"
                            : "#9e9e9e"
                        }`,
                        backgroundColor:
                          currentTurnData?.player_id === player.id
                            ? "action.selected"
                            : "transparent",
                      }}
                    >
                      <Box
                        sx={{
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                        }}
                      >
                        <Typography variant="subtitle2">
                          {player.model_name || player.type} ({player.color})
                        </Typography>
                        <Typography variant="body2">
                          VP: {pState.victory_points_visible}
                          {pState.has_longest_road && " 🛤️"}
                          {pState.has_largest_army && " ⚔️"}
                        </Typography>
                      </Box>

                      {pState.handVisible && pState.hand_resources && (
                        <Box sx={{ mt: 1 }}>
                          <Typography variant="caption" color="text.secondary">
                            Hand:{" "}
                            {Object.entries(pState.hand_resources)
                              .filter(([_, count]) => count > 0)
                              .map(([res, count]) => `${res}:${count}`)
                              .join(", ") || "empty"}
                          </Typography>
                        </Box>
                      )}

                      {!pState.handVisible && (
                        <Typography
                          variant="caption"
                          color="text.secondary"
                          sx={{ display: "block", mt: 0.5 }}
                        >
                          Cards: {pState.resource_count} resources,{" "}
                          {pState.dev_card_count} dev cards
                        </Typography>
                      )}
                    </Paper>
                  );
                })}
              </Box>

              {/* Robber location */}
              {visibleState.board?.robber_coordinate && (
                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{ display: "block", mt: 2 }}
                >
                  Robber at: {visibleState.board.robber_coordinate.join(", ")}
                </Typography>
              )}
            </Box>
          )}

          {/* Note about full board */}
          <Alert severity="info" sx={{ mt: 2 }}>
            Full board visualization coming soon. For now, use the existing
            Settlers of Catan replay viewer for graphical board display.
          </Alert>
        </Paper>

        {/* Right: Details panel */}
        <Box sx={{ flex: "1 1 400px", display: "flex", flexDirection: "column", gap: 2 }}>
          {/* Tabs for different views */}
          <Paper elevation={2}>
            <Tabs
              value={activeTab}
              onChange={(_, v) => setActiveTab(v)}
              variant="fullWidth"
            >
              <Tab label="LLM Decision" />
              <Tab label="Game Result" />
            </Tabs>

            <Box sx={{ p: 2 }}>
              {activeTab === 0 && currentTurnData && currentPlayer && (
                <LLMReasoningPanel
                  decision={currentTurnData.llm_decision}
                  action={currentTurnData.action}
                  playerColor={currentPlayer.color}
                  showReasoning={shouldShowReasoning}
                />
              )}

              {activeTab === 1 && (
                <Box>
                  {gameLog.result ? (
                    <>
                      <Typography variant="h6" sx={{ mb: 1 }}>
                        {gameLog.result.termination_reason === "victory"
                          ? `Winner: ${gameLog.result.winner_id}`
                          : `Game ended: ${gameLog.result.termination_reason}`}
                      </Typography>

                      <Typography variant="body2" sx={{ mb: 2 }}>
                        Final Scores:
                      </Typography>
                      {Object.entries(gameLog.result.final_scores)
                        .sort(([, a], [, b]) => b - a)
                        .map(([playerId, score]) => {
                          const player = gameLog.players.find(
                            (p) => p.id === playerId
                          );
                          return (
                            <Typography key={playerId} variant="body2">
                              {player?.model_name || player?.type || playerId}:{" "}
                              {score} VP
                            </Typography>
                          );
                        })}

                      <Divider sx={{ my: 2 }} />

                      <Typography variant="subtitle2" sx={{ mb: 1 }}>
                        Statistics
                      </Typography>
                      <Typography variant="body2">
                        Total turns: {gameLog.result.total_turns}
                      </Typography>
                      <Typography variant="body2">
                        LLM calls: {gameLog.result.statistics.total_llm_calls}
                      </Typography>
                      <Typography variant="body2">
                        Tokens used:{" "}
                        {gameLog.result.statistics.total_tokens_used.toLocaleString()}
                      </Typography>
                      <Typography variant="body2">
                        Avg decision time:{" "}
                        {gameLog.result.statistics.average_decision_time_ms.toFixed(
                          0
                        )}
                        ms
                      </Typography>
                    </>
                  ) : (
                    <Typography variant="body2" color="text.secondary">
                      Game still in progress or no result data.
                    </Typography>
                  )}
                </Box>
              )}
            </Box>
          </Paper>

          {/* Playback controls */}
          <PlaybackControls
            currentTurn={currentTurn}
            totalTurns={totalTurns}
            isPlaying={isPlaying}
            playbackSpeed={playbackSpeed}
            onPlay={play}
            onPause={pause}
            onSeek={seekToTurn}
            onNext={nextTurn}
            onPrev={prevTurn}
            onSpeedChange={setPlaybackSpeed}
          />
        </Box>
      </Box>

      {/* Turn timeline */}
      <Box sx={{ mt: 2 }}>
        <TurnTimeline
          turns={gameLog.turns}
          players={gameLog.players}
          currentTurn={currentTurn}
          onSeek={seekToTurn}
        />
      </Box>

      {/* Keyboard shortcuts help */}
      <Paper elevation={1} sx={{ p: 1, mt: 2 }}>
        <Typography variant="caption" color="text.secondary">
          Keyboard shortcuts: Space/K = Play/Pause | ←/J = Previous | →/L = Next
          | Home = Start | End = Finish
        </Typography>
      </Paper>
    </Container>
  );
}

// Main component with provider wrapper
function ArenaReplayScreen() {
  const { gameId } = useParams();
  const [searchParams] = useSearchParams();
  const [logData, setLogData] = useState<ArenaGameLog | undefined>();
  const [logUrl, setLogUrl] = useState<string | undefined>();

  // Get log URL from params or default location
  useEffect(() => {
    const urlParam = searchParams.get("url");
    if (urlParam) {
      setLogUrl(urlParam);
    } else if (gameId) {
      // Try default location
      setLogUrl(`/game_logs/${gameId}.json`);
    }
  }, [gameId, searchParams]);

  return (
    <ArenaLogProvider logData={logData} logUrl={logUrl}>
      <ArenaReplayContent />
    </ArenaLogProvider>
  );
}

export default ArenaReplayScreen;
