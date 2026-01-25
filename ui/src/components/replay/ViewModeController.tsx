/**
 * View Mode Controller - Fog of War toggle
 *
 * Allows users to switch between:
 * - Omniscient: See all players' hands and reasoning
 * - Player POV: See only what one player saw
 * - Spectator: Public information only
 */

import React from "react";
import {
  Box,
  ToggleButtonGroup,
  ToggleButton,
  Select,
  MenuItem,
  Typography,
  Tooltip,
  FormControl,
  InputLabel,
} from "@mui/material";
import type { SelectChangeEvent } from "@mui/material/Select";
import VisibilityIcon from "@mui/icons-material/Visibility";
import VisibilityOffIcon from "@mui/icons-material/VisibilityOff";
import PersonIcon from "@mui/icons-material/Person";

import type { ViewMode, ArenaPlayer } from "../../utils/arenaApi.types";

interface ViewModeControllerProps {
  viewMode: ViewMode;
  onViewModeChange: (mode: ViewMode) => void;
  players: ArenaPlayer[];
}

const COLOR_MAP: Record<string, string> = {
  RED: "#e53935",
  BLUE: "#1e88e5",
  ORANGE: "#fb8c00",
  WHITE: "#9e9e9e",
};

export function ViewModeController({
  viewMode,
  onViewModeChange,
  players,
}: ViewModeControllerProps) {
  const handleModeChange = (
    _event: React.MouseEvent<HTMLElement>,
    value: string | null
  ) => {
    if (!value) return;

    if (value === "player_pov") {
      onViewModeChange({
        type: "player_pov",
        player_id: players[0]?.id,
      });
    } else {
      onViewModeChange({ type: value as "omniscient" | "spectator" });
    }
  };

  const handlePlayerChange = (event: SelectChangeEvent<string>) => {
    onViewModeChange({
      type: "player_pov",
      player_id: event.target.value,
    });
  };

  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "center",
        gap: 2,
        p: 1,
        flexWrap: "wrap",
      }}
    >
      <Typography variant="subtitle2" sx={{ fontWeight: "bold" }}>
        View Mode:
      </Typography>

      <ToggleButtonGroup
        value={viewMode.type}
        exclusive
        onChange={handleModeChange}
        size="small"
        aria-label="view mode"
      >
        <Tooltip title="See all players' hands and LLM reasoning">
          <ToggleButton value="omniscient" aria-label="omniscient mode">
            <VisibilityIcon sx={{ mr: 0.5 }} fontSize="small" />
            Omniscient
          </ToggleButton>
        </Tooltip>

        <Tooltip title="See only what one player could see">
          <ToggleButton value="player_pov" aria-label="player pov mode">
            <PersonIcon sx={{ mr: 0.5 }} fontSize="small" />
            Player View
          </ToggleButton>
        </Tooltip>

        <Tooltip title="Public information only (like spectating)">
          <ToggleButton value="spectator" aria-label="spectator mode">
            <VisibilityOffIcon sx={{ mr: 0.5 }} fontSize="small" />
            Spectator
          </ToggleButton>
        </Tooltip>
      </ToggleButtonGroup>

      {viewMode.type === "player_pov" && (
        <FormControl size="small" sx={{ minWidth: 150 }}>
          <InputLabel id="player-select-label">Player</InputLabel>
          <Select
            labelId="player-select-label"
            value={viewMode.player_id || players[0]?.id || ""}
            onChange={handlePlayerChange}
            label="Player"
          >
            {players.map((player) => (
              <MenuItem key={player.id} value={player.id}>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <Box
                    sx={{
                      width: 12,
                      height: 12,
                      borderRadius: "50%",
                      backgroundColor: COLOR_MAP[player.color] || "#888",
                      border: "1px solid rgba(0,0,0,0.2)",
                    }}
                  />
                  <span>
                    {player.model_name || player.type} ({player.color})
                  </span>
                </Box>
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      )}
    </Box>
  );
}

export default ViewModeController;
