import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import {
  Box,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Paper,
  IconButton,
  Slider,
} from '@mui/material';
import {
  PlayArrow,
  Pause,
  SkipPrevious,
  SkipNext,
  FastRewind,
  FastForward,
} from '@mui/icons-material';
import { CodenamesBoard } from '../components/codenames';
import './CodenamesReplayScreen.scss';

// Types for Codenames game log
interface CodenamesPlayer {
  id: string;
  team: 'red' | 'blue';
  role: 'spymaster' | 'operative';
  type: string;
  model?: string;
}

interface CodenamesTurn {
  turn_number: number;
  player_id: string;
  role: string;
  action: {
    action_type: string;
    clue?: string;
    number?: number;
    word?: string;
  };
  action_result?: {
    card_type?: string;
    correct?: boolean;
    game_over?: boolean;
    winner?: string;
  };
}

interface CodenamesGameLog {
  game_id: string;
  game_type: string;
  created_at: string;
  players: CodenamesPlayer[];
  initial_state?: {
    grid: string[][];
    card_types: Record<string, string>;
  };
  result: {
    winner?: string;
    final_scores?: Record<string, any>;
    total_turns: number;
    termination_reason: string;
  };
  turns: CodenamesTurn[];
}

// Sample data for demo
const SAMPLE_GAME: CodenamesGameLog = {
  game_id: 'sample-001',
  game_type: 'codenames',
  created_at: new Date().toISOString(),
  players: [
    { id: 'red_spy', team: 'red', role: 'spymaster', type: 'llm', model: 'gpt-4o' },
    { id: 'red_op1', team: 'red', role: 'operative', type: 'llm', model: 'claude-3' },
    { id: 'blue_spy', team: 'blue', role: 'spymaster', type: 'llm', model: 'gemini' },
    { id: 'blue_op1', team: 'blue', role: 'operative', type: 'llm', model: 'gpt-4' },
  ],
  initial_state: {
    grid: [
      ['APPLE', 'BERLIN', 'CAT', 'DRAGON', 'EAGLE'],
      ['FIRE', 'GHOST', 'HOTEL', 'ICE', 'JUNGLE'],
      ['KING', 'LEMON', 'MOON', 'NINJA', 'OCEAN'],
      ['PIANO', 'QUEEN', 'ROBOT', 'SNAKE', 'TOWER'],
      ['UNICORN', 'VAMPIRE', 'WHALE', 'XRAY', 'ZEBRA'],
    ],
    card_types: {
      'APPLE': 'red', 'BERLIN': 'blue', 'CAT': 'red', 'DRAGON': 'bystander', 'EAGLE': 'red',
      'FIRE': 'blue', 'GHOST': 'assassin', 'HOTEL': 'red', 'ICE': 'blue', 'JUNGLE': 'bystander',
      'KING': 'red', 'LEMON': 'bystander', 'MOON': 'blue', 'NINJA': 'red', 'OCEAN': 'blue',
      'PIANO': 'bystander', 'QUEEN': 'red', 'ROBOT': 'blue', 'SNAKE': 'bystander', 'TOWER': 'red',
      'UNICORN': 'blue', 'VAMPIRE': 'bystander', 'WHALE': 'red', 'XRAY': 'blue', 'ZEBRA': 'bystander',
    },
  },
  result: {
    winner: 'red',
    total_turns: 12,
    termination_reason: 'victory',
  },
  turns: [
    { turn_number: 1, player_id: 'red_spy', role: 'spymaster', action: { action_type: 'GIVE_CLUE', clue: 'FRUIT', number: 2 } },
    { turn_number: 2, player_id: 'red_op1', role: 'operative', action: { action_type: 'GUESS', word: 'APPLE' }, action_result: { card_type: 'red', correct: true } },
    { turn_number: 3, player_id: 'red_op1', role: 'operative', action: { action_type: 'GUESS', word: 'LEMON' }, action_result: { card_type: 'bystander', correct: false } },
    { turn_number: 4, player_id: 'blue_spy', role: 'spymaster', action: { action_type: 'GIVE_CLUE', clue: 'WATER', number: 3 } },
    { turn_number: 5, player_id: 'blue_op1', role: 'operative', action: { action_type: 'GUESS', word: 'OCEAN' }, action_result: { card_type: 'blue', correct: true } },
    { turn_number: 6, player_id: 'blue_op1', role: 'operative', action: { action_type: 'GUESS', word: 'ICE' }, action_result: { card_type: 'blue', correct: true } },
  ],
};

type ViewMode = 'spymaster' | 'operative' | 'spectator';

const CodenamesReplayScreen: React.FC = () => {
  const { gameId } = useParams<{ gameId?: string }>();
  const [gameLog, setGameLog] = useState<CodenamesGameLog | null>(null);
  const [currentTurnIndex, setCurrentTurnIndex] = useState(0);
  const [viewMode, setViewMode] = useState<ViewMode>('spectator');
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);

  // Load game log (demo uses sample data)
  useEffect(() => {
    // In production, fetch from API using gameId
    setGameLog(SAMPLE_GAME);
  }, [gameId]);

  // Auto-play functionality
  useEffect(() => {
    if (!isPlaying || !gameLog) return;

    const interval = setInterval(() => {
      setCurrentTurnIndex((prev) => {
        if (prev >= gameLog.turns.length - 1) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 2000 / playbackSpeed);

    return () => clearInterval(interval);
  }, [isPlaying, playbackSpeed, gameLog]);

  if (!gameLog) {
    return (
      <Box className="codenames-replay-screen loading">
        <Typography>Loading game...</Typography>
      </Box>
    );
  }

  // Compute current game state based on turns
  const computeCurrentState = () => {
    if (!gameLog.initial_state) return null;

    const revealed = new Set<string>();
    let currentTeam: 'red' | 'blue' = 'red';
    let currentRole: 'spymaster' | 'operative' = 'spymaster';
    let currentClue: { word: string; number: number } | null = null;
    let redRemaining = Object.values(gameLog.initial_state.card_types).filter(t => t === 'red').length;
    let blueRemaining = Object.values(gameLog.initial_state.card_types).filter(t => t === 'blue').length;
    let gameOver = false;
    let winner: string | null = null;

    // Apply turns up to current index
    for (let i = 0; i <= currentTurnIndex && i < gameLog.turns.length; i++) {
      const turn = gameLog.turns[i];

      if (turn.action.action_type === 'GIVE_CLUE') {
        currentClue = { word: turn.action.clue!, number: turn.action.number! };
        currentTeam = turn.player_id.includes('red') ? 'red' : 'blue';
        currentRole = 'operative';
      } else if (turn.action.action_type === 'GUESS' && turn.action.word) {
        revealed.add(turn.action.word);
        const cardType = gameLog.initial_state.card_types[turn.action.word];

        if (cardType === 'red') redRemaining--;
        if (cardType === 'blue') blueRemaining--;

        if (cardType === 'assassin') {
          gameOver = true;
          winner = turn.player_id.includes('red') ? 'blue' : 'red';
        } else if (redRemaining === 0) {
          gameOver = true;
          winner = 'red';
        } else if (blueRemaining === 0) {
          gameOver = true;
          winner = 'blue';
        } else if (!turn.action_result?.correct) {
          // Wrong guess - switch teams
          currentTeam = currentTeam === 'red' ? 'blue' : 'red';
          currentRole = 'spymaster';
          currentClue = null;
        }
      } else if (turn.action.action_type === 'PASS') {
        currentTeam = currentTeam === 'red' ? 'blue' : 'red';
        currentRole = 'spymaster';
        currentClue = null;
      }
    }

    // Build grid state
    const grid = gameLog.initial_state.grid.map(row =>
      row.map(word => ({
        word,
        revealed: revealed.has(word),
        card_type: revealed.has(word) ? gameLog.initial_state!.card_types[word] : undefined,
      }))
    );

    return {
      grid,
      cardTypes: viewMode !== 'operative' ? gameLog.initial_state.card_types : undefined,
      currentTeam,
      currentRole,
      currentClue,
      redRemaining,
      blueRemaining,
      gameOver,
      winner,
    };
  };

  const currentState = computeCurrentState();
  const currentTurn = gameLog.turns[currentTurnIndex];

  return (
    <Box className="codenames-replay-screen">
      <Box className="header">
        <Typography variant="h4">Codenames Replay</Typography>
        <Typography variant="subtitle1" color="textSecondary">
          Game: {gameLog.game_id.substring(0, 8)}...
        </Typography>
      </Box>

      <Box className="main-content">
        {/* Game Board */}
        <Box className="board-container">
          {currentState && (
            <CodenamesBoard
              grid={currentState.grid}
              cardTypes={currentState.cardTypes}
              currentClue={currentState.currentClue}
              currentTeam={currentState.currentTeam}
              currentRole={currentState.currentRole}
              viewMode={viewMode}
              redRemaining={currentState.redRemaining}
              blueRemaining={currentState.blueRemaining}
              gameOver={currentState.gameOver}
              winner={currentState.winner}
            />
          )}
        </Box>

        {/* Side Panel */}
        <Box className="side-panel">
          {/* View Mode Selector */}
          <Paper className="control-panel">
            <Typography variant="h6">View Mode</Typography>
            <FormControl fullWidth size="small">
              <Select
                value={viewMode}
                onChange={(e) => setViewMode(e.target.value as ViewMode)}
              >
                <MenuItem value="spectator">Spectator (See All)</MenuItem>
                <MenuItem value="spymaster">Spymaster View</MenuItem>
                <MenuItem value="operative">Operative View</MenuItem>
              </Select>
            </FormControl>
          </Paper>

          {/* Current Turn Info */}
          <Paper className="turn-info-panel">
            <Typography variant="h6">Turn {currentTurnIndex + 1}</Typography>
            {currentTurn && (
              <>
                <Typography>
                  <strong>Player:</strong> {currentTurn.player_id}
                </Typography>
                <Typography>
                  <strong>Role:</strong> {currentTurn.role}
                </Typography>
                <Typography>
                  <strong>Action:</strong> {currentTurn.action.action_type}
                </Typography>
                {currentTurn.action.clue && (
                  <Typography>
                    <strong>Clue:</strong> "{currentTurn.action.clue}" for {currentTurn.action.number}
                  </Typography>
                )}
                {currentTurn.action.word && (
                  <Typography>
                    <strong>Guess:</strong> {currentTurn.action.word}
                    {currentTurn.action_result && (
                      <span> ({currentTurn.action_result.card_type})</span>
                    )}
                  </Typography>
                )}
              </>
            )}
          </Paper>

          {/* Players */}
          <Paper className="players-panel">
            <Typography variant="h6">Players</Typography>
            {gameLog.players.map((player) => (
              <Box key={player.id} className={`player-item ${player.team}`}>
                <span className="player-id">{player.id}</span>
                <span className="player-role">{player.role}</span>
                {player.model && <span className="player-model">{player.model}</span>}
              </Box>
            ))}
          </Paper>
        </Box>
      </Box>

      {/* Playback Controls */}
      <Paper className="playback-controls">
        <IconButton onClick={() => setCurrentTurnIndex(0)}>
          <SkipPrevious />
        </IconButton>
        <IconButton onClick={() => setCurrentTurnIndex(Math.max(0, currentTurnIndex - 1))}>
          <FastRewind />
        </IconButton>
        <IconButton onClick={() => setIsPlaying(!isPlaying)}>
          {isPlaying ? <Pause /> : <PlayArrow />}
        </IconButton>
        <IconButton onClick={() => setCurrentTurnIndex(Math.min(gameLog.turns.length - 1, currentTurnIndex + 1))}>
          <FastForward />
        </IconButton>
        <IconButton onClick={() => setCurrentTurnIndex(gameLog.turns.length - 1)}>
          <SkipNext />
        </IconButton>

        <Box className="turn-slider">
          <Slider
            value={currentTurnIndex}
            min={0}
            max={gameLog.turns.length - 1}
            onChange={(_, value) => setCurrentTurnIndex(value as number)}
            valueLabelDisplay="auto"
            valueLabelFormat={(v) => `Turn ${v + 1}`}
          />
        </Box>

        <Typography className="turn-counter">
          {currentTurnIndex + 1} / {gameLog.turns.length}
        </Typography>
      </Paper>
    </Box>
  );
};

export default CodenamesReplayScreen;
