/**
 * Catan Live Game Screen
 *
 * Manual play mode: Copy prompts to LLM, paste JSON responses back.
 * Uses the existing Board component for rendering.
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Paper,
  Typography,
  Button,
  CircularProgress,
  Alert,
  Chip,
  LinearProgress,
  TextField,
  IconButton,
  Tooltip,
  Collapse,
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import SendIcon from '@mui/icons-material/Send';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import { TransformWrapper, TransformComponent } from 'react-zoom-pan-pinch';

import Board from './Board';
import useWindowSize from '../utils/useWindowSize';
import type { GameState, GameAction, TileCoordinate } from '../utils/api.types';

import './CatanLiveScreen.scss';

interface GameMessage {
  type: string;
  [key: string]: any;
}

const CatanGameScreen: React.FC = () => {
  const { gameId } = useParams<{ gameId: string }>();
  const navigate = useNavigate();
  const wsRef = useRef<WebSocket | null>(null);
  const { width, height } = useWindowSize();

  const [connected, setConnected] = useState(false);
  const [gameStarted, setGameStarted] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Game state
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [players, setPlayers] = useState<any[]>([]);
  const [currentPlayer, setCurrentPlayer] = useState<string>('');
  const [currentColor, setCurrentColor] = useState<string>('');
  const [llmThinking, setLlmThinking] = useState<any>(null);
  const [lastAction, setLastAction] = useState<any>(null);
  const [gameOver, setGameOver] = useState(false);
  const [winner, setWinner] = useState<string | null>(null);
  const [turnNumber, setTurnNumber] = useState(0);
  const [showBoard, setShowBoard] = useState(false);

  // Manual play state
  const [waitingForInput, setWaitingForInput] = useState(false);
  const [currentPrompt, setCurrentPrompt] = useState<string>('');
  const [systemPrompt, setSystemPrompt] = useState<string>('');
  const [llmResponse, setLlmResponse] = useState<string>('');
  const [promptExpanded, setPromptExpanded] = useState(true);
  const [copySuccess, setCopySuccess] = useState(false);

  // Connect to WebSocket
  useEffect(() => {
    if (!gameId) return;

    // Use the current host to go through Vite's proxy, or connect directly to backend
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws/arena/${gameId}`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnected(true);
      setError(null); // Clear any previous errors
    };

    ws.onmessage = (event) => {
      const msg: GameMessage = JSON.parse(event.data);
      handleMessage(msg);
    };

    ws.onerror = (err) => {
      console.error('WebSocket error:', err);
      // Only set error if we never connected
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        // Don't set error immediately - wait to see if reconnect succeeds
      }
    };

    ws.onclose = () => {
      console.log('WebSocket closed');
      setConnected(false);
    };

    return () => {
      // Mark that we're intentionally closing
      wsRef.current = null;
      ws.close();
    };
  }, [gameId]);

  // Show board with delay for animation
  useEffect(() => {
    if (gameState) {
      setTimeout(() => setShowBoard(true), 300);
    }
  }, [gameState]);

  const handleMessage = useCallback((msg: GameMessage) => {
    console.log('WS message:', msg.type, msg);

    switch (msg.type) {
      case 'connected':
        setGameState(msg.initial_state);
        setPlayers(msg.players || []);
        break;

      case 'game_started':
        setGameStarted(true);
        break;

      case 'turn_start':
        setTurnNumber(msg.turn_number);
        setCurrentPlayer(msg.current_player);
        setCurrentColor(msg.current_color);
        if (msg.game_state) {
          setGameState(msg.game_state);
        }
        // Store prompt for manual play
        if (msg.prompt) {
          setCurrentPrompt(msg.prompt);
          setSystemPrompt(msg.system_prompt || '');
        }
        setLlmThinking(null);
        setLastAction(null);
        setWaitingForInput(false);
        break;

      case 'waiting_for_input':
        setWaitingForInput(true);
        setLlmResponse(''); // Clear previous response
        break;

      case 'llm_thinking':
        setLlmThinking({
          player_id: msg.player_id,
          model: msg.model,
          status: 'thinking',
        });
        setWaitingForInput(false);
        break;

      case 'llm_decision':
        setLlmThinking({
          player_id: msg.player_id,
          action: msg.action,
          reasoning: msg.reasoning,
          status: 'decided',
        });
        break;

      case 'action_executed':
        if (msg.game_state) {
          setGameState(msg.game_state);
        }
        setLastAction(msg);
        setLlmThinking(null);
        setWaitingForInput(false);
        if (msg.game_over) {
          setGameOver(true);
        }
        break;

      case 'game_finished':
        setGameOver(true);
        setWinner(msg.winner);
        setWaitingForInput(false);
        break;

      case 'error':
        setError(msg.message);
        break;
    }
  }, []);

  const startGame = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'start' }));
    }
  };

  const handleBack = () => {
    navigate('/');
  };

  const copyPromptToClipboard = async () => {
    const fullPrompt = `SYSTEM:\n${systemPrompt}\n\nUSER:\n${currentPrompt}`;
    try {
      await navigator.clipboard.writeText(fullPrompt);
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const submitAction = () => {
    if (!llmResponse.trim()) {
      setError('Please paste the LLM response');
      return;
    }

    try {
      // Try to parse the JSON response
      let parsed;

      // Try to extract JSON from markdown code blocks
      const jsonMatch = llmResponse.match(/```(?:json)?\s*([\s\S]*?)```/);
      if (jsonMatch) {
        parsed = JSON.parse(jsonMatch[1].trim());
      } else {
        // Try direct parse
        parsed = JSON.parse(llmResponse.trim());
      }

      // Send the action to the server
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          action: 'submit_action',
          ...parsed,
        }));
        setLlmResponse('');
        setError(null);
      }
    } catch (err) {
      setError('Invalid JSON response. Please check the format.');
    }
  };

  // No-op handlers for replay mode (LLMs control the game)
  const buildOnNodeClick = useCallback(
    (id: number, _action?: GameAction) => () => {
      // In live mode, we just watch - no user interaction
      console.log('Node clicked (spectator mode):', id);
    },
    []
  );

  const buildOnEdgeClick = useCallback(
    (id: [number, number], _action?: GameAction) => () => {
      // In live mode, we just watch - no user interaction
      console.log('Edge clicked (spectator mode):', id);
    },
    []
  );

  const handleTileClick = useCallback((coordinate: TileCoordinate) => {
    // In live mode, we just watch - no user interaction
    console.log('Tile clicked (spectator mode):', coordinate);
  }, []);

  if (!gameId) {
    return (
      <Box className="catan-live-screen loading">
        <Typography>No game ID provided</Typography>
        <Button onClick={handleBack}>Back to Home</Button>
      </Box>
    );
  }

  if (!connected && error) {
    return (
      <Box className="catan-live-screen" sx={{ p: 4 }}>
        <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>
        <Button variant="contained" onClick={handleBack}>Back to Home</Button>
      </Box>
    );
  }

  if (!connected) {
    return (
      <Box className="catan-live-screen loading">
        <CircularProgress size={60} />
        <Typography variant="h6" sx={{ mt: 2 }}>
          Connecting to game {gameId}...
        </Typography>
      </Box>
    );
  }

  // Get player color styles
  const getColorStyle = (color: string) => {
    const colorMap: Record<string, string> = {
      'RED': '#dc3545',
      'BLUE': '#007bff',
      'ORANGE': '#fd7e14',
      'WHITE': '#6c757d',
    };
    return colorMap[color] || '#333';
  };

  return (
    <Box className="catan-live-screen">
      {/* Header */}
      <Paper className="header" elevation={2}>
        <Button startIcon={<ArrowBackIcon />} onClick={handleBack} sx={{ color: 'white' }}>
          Back
        </Button>
        <Typography variant="h5">
          Settlers of Catan - Game {gameId}
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Chip
            label={gameOver ? 'Game Over' : `Turn ${turnNumber}`}
            variant="outlined"
            sx={{ color: 'white', borderColor: 'rgba(255,255,255,0.5)' }}
          />
          {currentColor && (
            <Chip
              label={currentColor}
              sx={{
                backgroundColor: getColorStyle(currentColor),
                color: 'white',
              }}
            />
          )}
        </Box>
      </Paper>

      {/* Error display */}
      {error && (
        <Alert severity="error" sx={{ m: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Pre-game: Start button */}
      {!gameStarted && !gameOver && (
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', p: 4 }}>
          <Typography variant="h6" sx={{ mb: 2, color: 'white' }}>
            Game Ready - Manual LLM Mode
          </Typography>
          <Typography variant="body2" sx={{ mb: 3, color: 'rgba(255,255,255,0.7)', textAlign: 'center', maxWidth: 500 }}>
            In this mode, you'll copy the game prompts to your LLM and paste back the JSON responses.
            The game will wait for your input at each turn.
          </Typography>
          <Button variant="contained" color="primary" size="large" onClick={startGame}>
            Start Game
          </Button>
        </Box>
      )}

      {/* Game in progress */}
      {(gameStarted || gameOver) && gameState && (
        <Box className="main-content" sx={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
          {/* Board */}
          <Box className="board-section" sx={{ flex: 1, position: 'relative' }}>
            <TransformWrapper>
              <div className="board-container">
                <TransformComponent>
                  {width && height && (
                    <Board
                      width={width - 420}
                      height={height - 144}
                      buildOnNodeClick={buildOnNodeClick}
                      buildOnEdgeClick={buildOnEdgeClick}
                      handleTileClick={handleTileClick}
                      nodeActions={{}}
                      edgeActions={{}}
                      replayMode={true}
                      show={showBoard}
                      gameState={gameState}
                      isMobile={false}
                      isMovingRobber={false}
                    />
                  )}
                </TransformComponent>
              </div>
            </TransformWrapper>
          </Box>

          {/* Side panel */}
          <Box className="side-panel" sx={{ width: 400, p: 2, overflowY: 'auto' }}>
            {/* Manual Play Section */}
            {waitingForInput && (
              <Paper elevation={2} sx={{ p: 2, mb: 2, backgroundColor: '#1e3a5f' }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6" sx={{ color: '#4fc3f7' }}>
                    {currentColor}'s Turn - Waiting for Input
                  </Typography>
                  <Chip label="MANUAL" size="small" sx={{ backgroundColor: '#4fc3f7', color: '#000' }} />
                </Box>

                {/* Prompt Section */}
                <Box sx={{ mb: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="subtitle2" sx={{ color: 'rgba(255,255,255,0.7)' }}>
                      LLM Prompt
                    </Typography>
                    <Box>
                      <Tooltip title={copySuccess ? "Copied!" : "Copy prompt"}>
                        <IconButton size="small" onClick={copyPromptToClipboard} sx={{ color: 'white' }}>
                          <ContentCopyIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                      <IconButton
                        size="small"
                        onClick={() => setPromptExpanded(!promptExpanded)}
                        sx={{ color: 'white' }}
                      >
                        {promptExpanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                      </IconButton>
                    </Box>
                  </Box>
                  <Collapse in={promptExpanded}>
                    <Paper
                      sx={{
                        p: 1.5,
                        mt: 1,
                        backgroundColor: '#0d1b2a',
                        maxHeight: 200,
                        overflow: 'auto',
                      }}
                    >
                      <Typography
                        variant="body2"
                        component="pre"
                        sx={{
                          fontFamily: 'monospace',
                          fontSize: '0.75rem',
                          color: 'rgba(255,255,255,0.9)',
                          whiteSpace: 'pre-wrap',
                          wordBreak: 'break-word',
                          m: 0,
                        }}
                      >
                        {currentPrompt}
                      </Typography>
                    </Paper>
                  </Collapse>
                </Box>

                {/* Response Input */}
                <Box>
                  <Typography variant="subtitle2" sx={{ color: 'rgba(255,255,255,0.7)', mb: 1 }}>
                    Paste LLM Response (JSON)
                  </Typography>
                  <TextField
                    multiline
                    rows={4}
                    fullWidth
                    value={llmResponse}
                    onChange={(e) => setLlmResponse(e.target.value)}
                    placeholder='{"action_type": "BUILD_SETTLEMENT", "value": 23, "rationale": "..."}'
                    sx={{
                      backgroundColor: '#0d1b2a',
                      '& .MuiInputBase-input': {
                        color: 'white',
                        fontFamily: 'monospace',
                        fontSize: '0.85rem',
                      },
                      '& .MuiOutlinedInput-root': {
                        '& fieldset': { borderColor: 'rgba(255,255,255,0.3)' },
                        '&:hover fieldset': { borderColor: 'rgba(255,255,255,0.5)' },
                        '&.Mui-focused fieldset': { borderColor: '#4fc3f7' },
                      },
                    }}
                  />
                  <Button
                    variant="contained"
                    fullWidth
                    onClick={submitAction}
                    startIcon={<SendIcon />}
                    sx={{ mt: 2, backgroundColor: '#4fc3f7', color: '#000' }}
                  >
                    Submit Move
                  </Button>
                </Box>
              </Paper>
            )}

            {/* LLM Status (for auto players) */}
            {llmThinking && !waitingForInput && (
              <Paper className="llm-section" elevation={2} sx={{ p: 2, mb: 2, backgroundColor: '#2c3e50' }}>
                <Typography variant="h6" gutterBottom sx={{ color: 'white' }}>
                  LLM Status
                </Typography>
                <Box className="llm-thinking">
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="subtitle2" sx={{ color: 'white' }}>
                      {llmThinking.model || currentPlayer}
                    </Typography>
                    <Chip
                      label={llmThinking.status}
                      size="small"
                      color={llmThinking.status === 'thinking' ? 'warning' : 'success'}
                    />
                  </Box>
                  {llmThinking.status === 'thinking' && <LinearProgress />}
                  {llmThinking.reasoning && (
                    <Box className="reasoning" sx={{ mt: 1 }}>
                      <Typography variant="caption" color="text.secondary">Reasoning:</Typography>
                      <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.9)' }}>
                        {llmThinking.reasoning}
                      </Typography>
                    </Box>
                  )}
                  {llmThinking.action && (
                    <Box className="action" sx={{ mt: 1 }}>
                      <Typography variant="caption" color="text.secondary">Action:</Typography>
                      <Typography variant="body2" sx={{ fontFamily: 'monospace', color: 'rgba(255,255,255,0.9)' }}>
                        {JSON.stringify(llmThinking.action)}
                      </Typography>
                    </Box>
                  )}
                </Box>
              </Paper>
            )}

            {/* Players */}
            <Paper className="teams-section" elevation={2} sx={{ p: 2, mb: 2, backgroundColor: '#2c3e50' }}>
              <Typography variant="h6" gutterBottom sx={{ color: 'white' }}>
                Players
              </Typography>
              {players.map((player: any, i: number) => {
                const colors = ['RED', 'BLUE', 'ORANGE', 'WHITE'];
                const playerColor = colors[i] || 'WHITE';
                const isCurrentPlayer = currentColor === playerColor;
                return (
                  <Box
                    key={i}
                    sx={{
                      p: 1,
                      mb: 1,
                      borderRadius: 1,
                      backgroundColor: `${getColorStyle(playerColor)}33`,
                      borderLeft: `3px solid ${getColorStyle(playerColor)}`,
                      opacity: isCurrentPlayer ? 1 : 0.7,
                    }}
                  >
                    <Typography variant="body2" sx={{ color: 'white' }}>
                      {playerColor}: {player.type === 'manual' ? 'Manual (You)' : player.model || player.type}
                    </Typography>
                  </Box>
                );
              })}
            </Paper>

            {/* Game Status */}
            {gameState && (
              <Paper sx={{ p: 2, mb: 2, backgroundColor: '#2c3e50' }}>
                <Typography variant="h6" gutterBottom sx={{ color: 'white' }}>
                  Game Status
                </Typography>
                <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.8)' }}>
                  Phase: {gameState.is_initial_build_phase ? 'Initial Placement' : 'Main Game'}
                </Typography>
                <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.8)' }}>
                  Prompt: {gameState.current_prompt}
                </Typography>
                {gameState.winning_color && (
                  <Typography variant="body2" sx={{ color: getColorStyle(gameState.winning_color), fontWeight: 'bold', mt: 1 }}>
                    Winner: {gameState.winning_color}
                  </Typography>
                )}
              </Paper>
            )}

            {/* Game Over */}
            {gameOver && winner && (
              <Paper sx={{ p: 2, backgroundColor: getColorStyle(winner) }}>
                <Typography variant="h6" sx={{ color: 'white', textAlign: 'center' }}>
                  {winner} WINS!
                </Typography>
              </Paper>
            )}
          </Box>
        </Box>
      )}
    </Box>
  );
};

export default CatanGameScreen;
