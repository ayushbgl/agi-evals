/**
 * Turn Timeline - Visual timeline of game turns
 *
 * Shows a scrollable timeline of turns with:
 * - Player color indicators
 * - Action type icons
 * - Current turn highlight
 * - Click-to-seek functionality
 */

import React, { useRef, useEffect } from "react";
import {
  Box,
  Paper,
  Typography,
  Tooltip,
  Chip,
} from "@mui/material";

import type { TurnRecord, ArenaPlayer } from "../../utils/arenaApi.types";

interface TurnTimelineProps {
  turns: TurnRecord[];
  players: ArenaPlayer[];
  currentTurn: number;
  onSeek: (turn: number) => void;
}

const COLOR_MAP: Record<string, string> = {
  RED: "#e53935",
  BLUE: "#1e88e5",
  ORANGE: "#fb8c00",
  WHITE: "#bdbdbd",
};

// Action type to icon/emoji mapping
const ACTION_ICONS: Record<string, string> = {
  ROLL: "🎲",
  END_TURN: "⏭️",
  BUILD_ROAD: "🛤️",
  BUILD_SETTLEMENT: "🏠",
  BUILD_CITY: "🏙️",
  BUY_DEVELOPMENT_CARD: "🃏",
  PLAY_KNIGHT_CARD: "⚔️",
  PLAY_YEAR_OF_PLENTY: "🌾",
  PLAY_MONOPOLY: "💰",
  PLAY_ROAD_BUILDING: "🛣️",
  MOVE_ROBBER: "🦹",
  MARITIME_TRADE: "⚓",
  OFFER_TRADE: "🤝",
  ACCEPT_TRADE: "✅",
  REJECT_TRADE: "❌",
  CONFIRM_TRADE: "📝",
  CANCEL_TRADE: "🚫",
  DISCARD: "🗑️",
};

export function TurnTimeline({
  turns,
  players,
  currentTurn,
  onSeek,
}: TurnTimelineProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const currentTurnRef = useRef<HTMLDivElement>(null);

  // Get player color by ID
  const getPlayerColor = (playerId: string): string => {
    const player = players.find((p) => p.id === playerId);
    return player ? COLOR_MAP[player.color] || "#888" : "#888";
  };

  // Get player name/model
  const getPlayerName = (playerId: string): string => {
    const player = players.find((p) => p.id === playerId);
    if (!player) return playerId;
    return player.model_name || player.type;
  };

  // Scroll to current turn when it changes
  useEffect(() => {
    if (currentTurnRef.current && containerRef.current) {
      const container = containerRef.current;
      const element = currentTurnRef.current;

      const containerRect = container.getBoundingClientRect();
      const elementRect = element.getBoundingClientRect();

      // Check if element is outside visible area
      if (
        elementRect.left < containerRect.left ||
        elementRect.right > containerRect.right
      ) {
        element.scrollIntoView({
          behavior: "smooth",
          block: "nearest",
          inline: "center",
        });
      }
    }
  }, [currentTurn]);

  // Group consecutive turns by player for cleaner display
  const groupedTurns = React.useMemo(() => {
    const groups: { playerId: string; startTurn: number; turns: TurnRecord[] }[] = [];
    let currentGroup: typeof groups[0] | null = null;

    turns.forEach((turn, idx) => {
      if (!currentGroup || currentGroup.playerId !== turn.player_id) {
        currentGroup = {
          playerId: turn.player_id,
          startTurn: idx,
          turns: [turn],
        };
        groups.push(currentGroup);
      } else {
        currentGroup.turns.push(turn);
      }
    });

    return groups;
  }, [turns]);

  return (
    <Paper elevation={2} sx={{ p: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: "bold" }}>
        Turn Timeline
      </Typography>

      {/* Legend */}
      <Box sx={{ display: "flex", gap: 1, mb: 2, flexWrap: "wrap" }}>
        {players.map((player) => (
          <Chip
            key={player.id}
            label={`${player.model_name || player.type} (${player.color})`}
            size="small"
            sx={{
              backgroundColor: COLOR_MAP[player.color] || "#888",
              color: player.color === "WHITE" ? "#000" : "#fff",
            }}
          />
        ))}
      </Box>

      {/* Timeline container */}
      <Box
        ref={containerRef}
        sx={{
          display: "flex",
          overflowX: "auto",
          gap: 0.5,
          pb: 1,
          "&::-webkit-scrollbar": {
            height: 8,
          },
          "&::-webkit-scrollbar-thumb": {
            backgroundColor: "grey.400",
            borderRadius: 4,
          },
        }}
      >
        {turns.map((turn, idx) => {
          const isCurrentTurn = idx === currentTurn;
          const playerColor = getPlayerColor(turn.player_id);
          const actionIcon = ACTION_ICONS[turn.action.action_type] || "▪️";

          return (
            <Tooltip
              key={idx}
              title={
                <Box>
                  <Typography variant="body2">
                    Turn {idx + 1}: {getPlayerName(turn.player_id)}
                  </Typography>
                  <Typography variant="body2">
                    {turn.action.action_type}
                  </Typography>
                  {turn.action.rationale && (
                    <Typography variant="caption" sx={{ fontStyle: "italic" }}>
                      "{turn.action.rationale}"
                    </Typography>
                  )}
                </Box>
              }
            >
              <Box
                ref={isCurrentTurn ? currentTurnRef : null}
                onClick={() => onSeek(idx)}
                sx={{
                  minWidth: 40,
                  height: 48,
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  borderRadius: 1,
                  cursor: "pointer",
                  backgroundColor: isCurrentTurn
                    ? "primary.main"
                    : "grey.100",
                  border: `2px solid ${isCurrentTurn ? "primary.dark" : playerColor}`,
                  borderTopWidth: 4,
                  borderTopColor: playerColor,
                  transition: "all 0.15s ease-in-out",
                  "&:hover": {
                    transform: "scale(1.1)",
                    zIndex: 1,
                    boxShadow: 2,
                  },
                }}
              >
                <Typography
                  sx={{
                    fontSize: "1.2rem",
                    lineHeight: 1,
                  }}
                >
                  {actionIcon}
                </Typography>
                <Typography
                  variant="caption"
                  sx={{
                    fontSize: "0.65rem",
                    color: isCurrentTurn ? "white" : "text.secondary",
                  }}
                >
                  {idx + 1}
                </Typography>
              </Box>
            </Tooltip>
          );
        })}
      </Box>

      {/* Summary */}
      <Box sx={{ mt: 1, display: "flex", justifyContent: "space-between" }}>
        <Typography variant="caption" color="text.secondary">
          {turns.length} total turns
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Click to jump to turn
        </Typography>
      </Box>
    </Paper>
  );
}

export default TurnTimeline;
