/**
 * Catan Live Game Screen
 *
 * Configuration and live view for Catan games with LLM players.
 * Requires backend arena server to be running.
 */

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Paper,
  Typography,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Alert,
  Chip,
  IconButton,
  Divider,
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';

import './CatanLiveScreen.scss';

// Available LLM models
const AVAILABLE_MODELS = [
  { id: 'gpt-4o', name: 'GPT-4o (OpenAI)', provider: 'openai' },
  { id: 'gpt-4-turbo', name: 'GPT-4 Turbo (OpenAI)', provider: 'openai' },
  { id: 'claude-3-opus', name: 'Claude 3 Opus (Anthropic)', provider: 'anthropic' },
  { id: 'claude-3-sonnet', name: 'Claude 3 Sonnet (Anthropic)', provider: 'anthropic' },
  { id: 'gemini-pro', name: 'Gemini Pro (Google)', provider: 'google' },
  { id: 'llama-3-70b', name: 'Llama 3 70B (Meta)', provider: 'meta' },
];

// Player types
const PLAYER_TYPES = [
  { id: 'manual', name: 'Manual (Copy/Paste LLM)' },
  { id: 'llm', name: 'Auto LLM (API Call)' },
  { id: 'random', name: 'Random Bot' },
];

const PLAYER_COLORS = ['RED', 'BLUE', 'ORANGE', 'WHITE'];

interface PlayerConfig {
  id: string;
  color: string;
  type: string;
  model: string;
}

const CatanLiveScreen: React.FC = () => {
  const navigate = useNavigate();
  const [players, setPlayers] = useState<PlayerConfig[]>([
    { id: 'player_1', color: 'RED', type: 'manual', model: 'gpt-4o' },
    { id: 'player_2', color: 'BLUE', type: 'manual', model: 'claude-3-opus' },
  ]);
  const [isStarting, setIsStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAddPlayer = () => {
    if (players.length >= 4) return;
    const usedColors = players.map(p => p.color);
    const nextColor = PLAYER_COLORS.find(c => !usedColors.includes(c)) || 'WHITE';
    setPlayers([
      ...players,
      { id: `player_${players.length + 1}`, color: nextColor, type: 'manual', model: 'gpt-4o' },
    ]);
  };

  const handleRemovePlayer = (index: number) => {
    if (players.length <= 2) return;
    setPlayers(players.filter((_, i) => i !== index));
  };

  const handlePlayerChange = (index: number, field: keyof PlayerConfig, value: string) => {
    const newPlayers = [...players];
    newPlayers[index] = { ...newPlayers[index], [field]: value };
    setPlayers(newPlayers);
  };

  const handleStartGame = async () => {
    setIsStarting(true);
    setError(null);

    try {
      const response = await fetch('/api/arena/catan/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          players: players.map(p => ({
            type: p.type,
            model: p.type !== 'random' ? p.model : undefined,
            color: p.color,
          })),
          config: {
            map_type: 'BASE',
            vps_to_win: 10,
            max_turns: 500,
          },
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to start game');
      }

      const data = await response.json();
      // Navigate to the game view with the game ID
      navigate(`/catan/live/${data.game_id}`);
    } catch (err) {
      setError(
        'Could not connect to the arena backend server. ' +
        'Make sure the server is running with: python -m arena.server'
      );
      setIsStarting(false);
    }
  };

  const handleBack = () => {
    navigate('/');
  };

  return (
    <Box className="catan-live-screen">
      {/* Header */}
      <Paper className="header" elevation={2}>
        <Button startIcon={<ArrowBackIcon />} onClick={handleBack} sx={{ color: 'white' }}>
          Back
        </Button>
        <Typography variant="h5">
          Settlers of Catan - LLM Arena
        </Typography>
        <Box sx={{ width: 100 }} />
      </Paper>

      <Container maxWidth="md" sx={{ py: 4 }}>
        {/* Info Alert */}
        <Alert severity="info" sx={{ mb: 3 }}>
          <Typography variant="body2">
            <strong>How it works:</strong> Configure players below, then start the game.
          </Typography>
          <Typography variant="body2" sx={{ mt: 1 }}>
            <strong>Manual mode:</strong> Copy the game prompt, paste it into your LLM (ChatGPT, Claude, etc.),
            then copy the JSON response back into the game.
          </Typography>
          <Typography variant="body2" sx={{ mt: 1 }}>
            <strong>Auto LLM:</strong> Requires API keys in the backend - the server will call LLM APIs automatically.
          </Typography>
        </Alert>

        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        {/* Player Configuration */}
        <Paper sx={{ p: 3, mb: 3, backgroundColor: '#2c3e50' }}>
          <Typography variant="h6" gutterBottom sx={{ color: 'white' }}>
            Configure Players
          </Typography>
          <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.7)', mb: 2 }}>
            Select which LLM model each player will use. You need API keys configured in the backend.
          </Typography>

          {players.map((player, index) => (
            <Paper
              key={player.id}
              sx={{
                p: 2,
                mb: 2,
                display: 'flex',
                alignItems: 'center',
                gap: 2,
                flexWrap: 'wrap',
                borderLeft: `4px solid ${player.color.toLowerCase()}`,
              }}
            >
              <Chip
                label={player.color}
                size="small"
                sx={{
                  backgroundColor: player.color.toLowerCase(),
                  color: player.color === 'WHITE' ? 'black' : 'white',
                  minWidth: 70,
                }}
              />

              <FormControl sx={{ minWidth: 180 }}>
                <InputLabel>Player Type</InputLabel>
                <Select
                  value={player.type}
                  label="Player Type"
                  onChange={(e) => handlePlayerChange(index, 'type', e.target.value)}
                >
                  {PLAYER_TYPES.map(type => (
                    <MenuItem key={type.id} value={type.id}>
                      {type.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              {player.type !== 'random' && (
                <FormControl sx={{ minWidth: 180 }}>
                  <InputLabel>LLM Model</InputLabel>
                  <Select
                    value={player.model}
                    label="LLM Model"
                    onChange={(e) => handlePlayerChange(index, 'model', e.target.value)}
                  >
                    {AVAILABLE_MODELS.map(model => (
                      <MenuItem key={model.id} value={model.id}>
                        {model.name}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              )}

              <FormControl sx={{ minWidth: 100 }}>
                <InputLabel>Color</InputLabel>
                <Select
                  value={player.color}
                  label="Color"
                  onChange={(e) => handlePlayerChange(index, 'color', e.target.value)}
                >
                  {PLAYER_COLORS.map(color => (
                    <MenuItem key={color} value={color} disabled={players.some(p => p.color === color && p.id !== player.id)}>
                      {color}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              <Box sx={{ flex: 1 }} />

              <IconButton
                onClick={() => handleRemovePlayer(index)}
                disabled={players.length <= 2}
                sx={{ color: 'rgba(255,255,255,0.5)' }}
              >
                <DeleteIcon />
              </IconButton>
            </Paper>
          ))}

          {players.length < 4 && (
            <Button
              startIcon={<AddIcon />}
              onClick={handleAddPlayer}
              variant="outlined"
              sx={{ color: 'white', borderColor: 'rgba(255,255,255,0.3)' }}
            >
              Add Player
            </Button>
          )}
        </Paper>

        {/* Game Settings */}
        <Paper sx={{ p: 3, mb: 3, backgroundColor: '#2c3e50' }}>
          <Typography variant="h6" gutterBottom sx={{ color: 'white' }}>
            Game Settings
          </Typography>

          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
            <TextField
              label="Victory Points to Win"
              type="number"
              defaultValue={10}
              InputProps={{ inputProps: { min: 5, max: 15 } }}
              sx={{ width: 180 }}
            />
            <TextField
              label="Max Turns"
              type="number"
              defaultValue={500}
              InputProps={{ inputProps: { min: 100, max: 1000 } }}
              sx={{ width: 180 }}
            />
          </Box>
        </Paper>

        {/* Start Button */}
        <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2 }}>
          <Button
            variant="contained"
            color="primary"
            size="large"
            startIcon={<PlayArrowIcon />}
            onClick={handleStartGame}
            disabled={isStarting}
            sx={{ minWidth: 200 }}
          >
            {isStarting ? 'Starting...' : 'Start Game'}
          </Button>
          <Button
            variant="outlined"
            onClick={() => navigate('/catan/replay')}
            sx={{ color: 'white', borderColor: 'rgba(255,255,255,0.5)' }}
          >
            View Past Replays
          </Button>
        </Box>

        <Divider sx={{ my: 4, borderColor: 'rgba(255,255,255,0.1)' }} />

        {/* Backend Setup Instructions */}
        <Paper sx={{ p: 3, backgroundColor: '#1e2a35' }}>
          <Typography variant="h6" gutterBottom sx={{ color: 'white' }}>
            Backend Setup Required
          </Typography>
          <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.7)', mb: 2 }}>
            To run live games, you need to start the arena backend server:
          </Typography>
          <Paper sx={{ p: 2, backgroundColor: '#0d1117', fontFamily: 'monospace', fontSize: 14 }}>
            <Typography component="pre" sx={{ color: '#58a6ff', m: 0 }}>
              # Set your API keys{'\n'}
              export OPENAI_API_KEY="sk-..."{'\n'}
              export ANTHROPIC_API_KEY="sk-ant-..."{'\n'}
              {'\n'}
              # Start the arena server{'\n'}
              python -m arena.server --port 8000
            </Typography>
          </Paper>
          <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.5)', mt: 2 }}>
            The server will handle LLM API calls and stream game updates to this UI.
          </Typography>
        </Paper>
      </Container>
    </Box>
  );
};

export default CatanLiveScreen;
