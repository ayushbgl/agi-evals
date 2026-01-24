/**
 * Playback Controls - Play/Pause/Seek for game replay
 *
 * Provides VCR-style controls for stepping through game turns
 * with variable speed playback.
 */

import React from "react";
import {
  Box,
  IconButton,
  Slider,
  Typography,
  ToggleButtonGroup,
  ToggleButton,
  Tooltip,
  Paper,
} from "@mui/material";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PauseIcon from "@mui/icons-material/Pause";
import SkipPreviousIcon from "@mui/icons-material/SkipPrevious";
import SkipNextIcon from "@mui/icons-material/SkipNext";
import FastRewindIcon from "@mui/icons-material/FastRewind";
import FastForwardIcon from "@mui/icons-material/FastForward";
import FirstPageIcon from "@mui/icons-material/FirstPage";
import LastPageIcon from "@mui/icons-material/LastPage";

interface PlaybackControlsProps {
  currentTurn: number;
  totalTurns: number;
  isPlaying: boolean;
  playbackSpeed: number;
  onPlay: () => void;
  onPause: () => void;
  onSeek: (turn: number) => void;
  onNext: () => void;
  onPrev: () => void;
  onSpeedChange: (speed: number) => void;
}

const SPEED_OPTIONS = [0.5, 1, 2, 4];

export function PlaybackControls({
  currentTurn,
  totalTurns,
  isPlaying,
  playbackSpeed,
  onPlay,
  onPause,
  onSeek,
  onNext,
  onPrev,
  onSpeedChange,
}: PlaybackControlsProps) {
  const handleSliderChange = (_event: Event, value: number | number[]) => {
    onSeek(value as number);
  };

  const handleSpeedChange = (
    _event: React.MouseEvent<HTMLElement>,
    value: number | null
  ) => {
    if (value !== null) {
      onSpeedChange(value);
    }
  };

  const goToStart = () => onSeek(0);
  const goToEnd = () => onSeek(totalTurns - 1);
  const skipBack = () => onSeek(Math.max(0, currentTurn - 10));
  const skipForward = () => onSeek(Math.min(totalTurns - 1, currentTurn + 10));

  // Format turn label
  const formatTurnLabel = (turn: number): string => {
    return `Turn ${turn + 1}`;
  };

  return (
    <Paper
      elevation={2}
      sx={{
        p: 2,
        display: "flex",
        flexDirection: "column",
        gap: 2,
      }}
    >
      {/* Turn slider */}
      <Box sx={{ px: 2 }}>
        <Slider
          value={currentTurn}
          min={0}
          max={Math.max(0, totalTurns - 1)}
          onChange={handleSliderChange}
          valueLabelDisplay="auto"
          valueLabelFormat={formatTurnLabel}
          marks={
            totalTurns <= 20
              ? Array.from({ length: totalTurns }, (_, i) => ({
                  value: i,
                  label: i % 5 === 0 ? `${i + 1}` : "",
                }))
              : undefined
          }
          sx={{ mt: 1 }}
        />
      </Box>

      {/* Main controls */}
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: 1,
        }}
      >
        {/* Go to start */}
        <Tooltip title="Go to start">
          <IconButton onClick={goToStart} disabled={currentTurn === 0}>
            <FirstPageIcon />
          </IconButton>
        </Tooltip>

        {/* Skip back 10 */}
        <Tooltip title="Back 10 turns">
          <IconButton onClick={skipBack} disabled={currentTurn === 0}>
            <FastRewindIcon />
          </IconButton>
        </Tooltip>

        {/* Previous turn */}
        <Tooltip title="Previous turn">
          <IconButton onClick={onPrev} disabled={currentTurn === 0}>
            <SkipPreviousIcon />
          </IconButton>
        </Tooltip>

        {/* Play/Pause */}
        <Tooltip title={isPlaying ? "Pause" : "Play"}>
          <IconButton
            onClick={isPlaying ? onPause : onPlay}
            color="primary"
            sx={{
              backgroundColor: "primary.main",
              color: "white",
              "&:hover": { backgroundColor: "primary.dark" },
              mx: 1,
            }}
            size="large"
          >
            {isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
          </IconButton>
        </Tooltip>

        {/* Next turn */}
        <Tooltip title="Next turn">
          <IconButton
            onClick={onNext}
            disabled={currentTurn >= totalTurns - 1}
          >
            <SkipNextIcon />
          </IconButton>
        </Tooltip>

        {/* Skip forward 10 */}
        <Tooltip title="Forward 10 turns">
          <IconButton
            onClick={skipForward}
            disabled={currentTurn >= totalTurns - 1}
          >
            <FastForwardIcon />
          </IconButton>
        </Tooltip>

        {/* Go to end */}
        <Tooltip title="Go to end">
          <IconButton
            onClick={goToEnd}
            disabled={currentTurn >= totalTurns - 1}
          >
            <LastPageIcon />
          </IconButton>
        </Tooltip>
      </Box>

      {/* Speed controls and turn counter */}
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <Typography variant="body2" color="text.secondary">
          Turn {currentTurn + 1} of {totalTurns}
        </Typography>

        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <Typography variant="body2" color="text.secondary">
            Speed:
          </Typography>
          <ToggleButtonGroup
            value={playbackSpeed}
            exclusive
            onChange={handleSpeedChange}
            size="small"
            aria-label="playback speed"
          >
            {SPEED_OPTIONS.map((speed) => (
              <ToggleButton key={speed} value={speed} aria-label={`${speed}x`}>
                {speed}x
              </ToggleButton>
            ))}
          </ToggleButtonGroup>
        </Box>
      </Box>
    </Paper>
  );
}

export default PlaybackControls;
