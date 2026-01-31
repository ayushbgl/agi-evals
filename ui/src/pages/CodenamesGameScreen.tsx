/**
 * Codenames Live Game Screen
 *
 * Connects to WebSocket and displays game as LLMs play.
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

import { CodenamesBoard } from '../components/codenames';
import './CodenamesLiveScreen.scss';

interface GameMessage {
  type: string;
  [key: string]: any;
}

const CodenamesGameScreen: React.FC = () => {
  const { gameId } = useParams<{ gameId: string }>();
  const navigate = useNavigate();
  const wsRef = useRef<WebSocket | null>(null);

  const [connected, setConnected] = useState(false);
  const [gameStarted, setGameStarted] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Game state
  const [publicState, setPublicState] = useState<any>(null);
  const [players, setPlayers] = useState<any[]>([]);
  const [currentPlayer, setCurrentPlayer] = useState<string>('');
  const [currentRole, setCurrentRole] = useState<string>('');
  const [llmThinking, setLlmThinking] = useState<any>(null);
  const [lastAction, setLastAction] = useState<any>(null);
  const [gameOver, setGameOver] = useState(false);
  const [winner, setWinner] = useState<string | null>(null);
  const [turnNumber, setTurnNumber] = useState(0);

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
      setError('WebSocket connection failed');
    };

    ws.onclose = () => {
      console.log('WebSocket closed');
      setConnected(false);
    };

    return () => {
      ws.close();
    };
  }, [gameId]);

  const handleMessage = useCallback((msg: GameMessage) => {
    console.log('WS message:', msg.type, msg);

    switch (msg.type) {
      case 'connected':
        setPublicState(msg.initial_state);
        setPlayers(msg.players || []);
        break;

      case 'game_started':
        setGameStarted(true);
        break;

      case 'turn_start':
        setTurnNumber(msg.turn_number);
        setCurrentPlayer(msg.current_player);
        setCurrentRole(msg.current_role);
        setPublicState(msg.public_state);
        setLlmThinking(null);
        setLastAction(null);
        setWaitingForInput(false);
        break;

      case 'waiting_for_input':
        setWaitingForInput(true);
        setCurrentPrompt(msg.prompt || '');
        setSystemPrompt(msg.system_prompt || '');
        setLlmResponse('');
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
        setPublicState(msg.public_state);
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
      let parsed;
      const jsonMatch = llmResponse.match(/```(?:json)?\s*([\s\S]*?)```/);
      if (jsonMatch) {
        parsed = JSON.parse(jsonMatch[1].trim());
      } else {
        parsed = JSON.parse(llmResponse.trim());
      }

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

  // Build grid for board
  const buildGrid = () => {
    if (!publicState?.grid) return [];
    return publicState.grid.map((row: any[]) =>
      row.map((cell: any) => ({
        word: cell.word,
        revealed: cell.revealed,
        card_type: cell.card_type,
      }))
    );
  };

  // Build card types map (for spectator view, show all)
  const buildCardTypes = () => {
    if (!publicState?.grid) return {};
    const types: Record<string, string> = {};
    // We don't have access to hidden card types from public state
    // In spectator mode, we'd need the server to send them
    return types;
  };

  if (!gameId) {
    return (
      <Box className="codenames-live-screen loading">
        <Typography>No game ID provided</Typography>
        <Button onClick={handleBack}>Back to Home</Button>
      </Box>
    );
  }

  if (error) {
    return (
      <Box className="codenames-live-screen" sx={{ p: 4 }}>
        <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>
        <Button variant="contained" onClick={handleBack}>Back to Home</Button>
      </Box>
    );
  }

  if (!connected) {
    return (
      <Box className="codenames-live-screen loading">
        <CircularProgress size={60} />
        <Typography variant="h6" sx={{ mt: 2 }}>
          Connecting to game {gameId}...
        </Typography>
      </Box>
    );
  }

  return (
    <Box className="codenames-live-screen">
      {/* Header */}
      <Paper className="header" elevation={2}>
        <Button startIcon={<ArrowBackIcon />} onClick={handleBack} sx={{ color: 'white' }}>
          Back
        </Button>
        <Typography variant="h5">
          Codenames - Game {gameId}
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Chip
            label={gameOver ? 'Game Over' : `Turn ${turnNumber}`}
            variant="outlined"
            sx={{ color: 'white', borderColor: 'rgba(255,255,255,0.5)' }}
          />
          {publicState && (
            <Chip
              label={`${publicState.current_team?.toUpperCase()} - ${currentRole}`}
              sx={{
                backgroundColor: publicState.current_team === 'red' ? '#dc3545' : '#007bff',
                color: 'white',
              }}
            />
          )}
        </Box>
      </Paper>

      {/* Pre-game: Start button */}
      {!gameStarted && !gameOver && (
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', p: 4 }}>
          <Typography variant="h6" sx={{ mb: 2 }}>
            Game Ready
          </Typography>
          <Typography variant="body2" sx={{ mb: 3, color: 'rgba(255,255,255,0.7)' }}>
            {players.length} players connected. Click Start to begin.
          </Typography>
          <Button variant="contained" color="primary" size="large" onClick={startGame}>
            Start Game
          </Button>
        </Box>
      )}

      {/* Game in progress */}
      {(gameStarted || gameOver) && publicState && (
        <Box className="main-content">
          {/* Board */}
          <Box className="board-section">
            <CodenamesBoard
              grid={buildGrid()}
              cardTypes={buildCardTypes()}
              currentClue={publicState.current_clue}
              currentTeam={publicState.current_team || 'red'}
              currentRole={currentRole as 'spymaster' | 'operative'}
              viewMode="spectator"
              redRemaining={publicState.red_remaining || 0}
              blueRemaining={publicState.blue_remaining || 0}
              gameOver={gameOver}
              winner={winner}
            />
          </Box>

          {/* Side panel */}
          <Box className="side-panel">
            {/* Manual Play Section */}
            {waitingForInput && (
              <Paper elevation={2} sx={{ p: 2, mb: 2, backgroundColor: '#1e3a5f' }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6" sx={{ color: '#4fc3f7' }}>
                    {currentPlayer} ({currentRole}) - Waiting for Input
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
                    placeholder='{"action_type": "GIVE_CLUE", "word": "ocean", "number": 3}'
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

            {/* LLM Status */}
            {!waitingForInput && <Paper className="llm-section" elevation={2}>
              <Typography variant="h6" gutterBottom>
                LLM Status
              </Typography>
              {llmThinking ? (
                <Box className="llm-thinking">
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="subtitle2">{llmThinking.model || currentPlayer}</Typography>
                    <Chip
                      label={llmThinking.status}
                      size="small"
                      color={llmThinking.status === 'thinking' ? 'warning' : 'success'}
                    />
                  </Box>
                  {llmThinking.status === 'thinking' && <LinearProgress />}
                  {llmThinking.reasoning && (
                    <Box className="reasoning">
                      <Typography variant="caption" color="text.secondary">Reasoning:</Typography>
                      <Typography variant="body2">{llmThinking.reasoning}</Typography>
                    </Box>
                  )}
                  {llmThinking.action && (
                    <Box className="action">
                      <Typography variant="caption" color="text.secondary">Action:</Typography>
                      <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                        {JSON.stringify(llmThinking.action)}
                      </Typography>
                    </Box>
                  )}
                </Box>
              ) : lastAction ? (
                <Box>
                  <Typography variant="body2">
                    Last: {lastAction.action?.action_type}
                    {lastAction.action?.word && ` - ${lastAction.action.word}`}
                    {lastAction.action?.clue && ` - "${lastAction.action.clue}" for ${lastAction.action.number}`}
                  </Typography>
                </Box>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  Waiting for turn...
                </Typography>
              )}
            </Paper>}

            {/* Players */}
            <Paper className="teams-section" elevation={2}>
              <Typography variant="h6" gutterBottom>
                Players
              </Typography>
              {players.map((player: any, i: number) => (
                <Box
                  key={i}
                  sx={{
                    p: 1,
                    mb: 1,
                    borderRadius: 1,
                    backgroundColor: player.team === 'red' ? 'rgba(220,53,69,0.2)' : 'rgba(0,123,255,0.2)',
                    borderLeft: `3px solid ${player.team === 'red' ? '#dc3545' : '#007bff'}`,
                    opacity: player.id === currentPlayer ? 1 : 0.7,
                  }}
                >
                  <Typography variant="body2">
                    {player.role}: {player.model || player.type}
                  </Typography>
                </Box>
              ))}
            </Paper>

            {/* Game Over */}
            {gameOver && (
              <Paper sx={{ p: 2, backgroundColor: winner === 'red' ? '#dc3545' : '#007bff' }}>
                <Typography variant="h6" sx={{ color: 'white', textAlign: 'center' }}>
                  {winner?.toUpperCase()} WINS!
                </Typography>
              </Paper>
            )}
          </Box>
        </Box>
      )}
    </Box>
  );
};

export default CodenamesGameScreen;
