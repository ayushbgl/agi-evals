/**
 * Codenames Live Game Screen
 *
 * Configuration and live view for Codenames games with LLM players.
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
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Alert,
  Chip,
  Divider,
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';

import './CodenamesLiveScreen.scss';

// Available LLM models
const AVAILABLE_MODELS = [
  { id: 'gpt-4o', name: 'GPT-4o (OpenAI)', provider: 'openai' },
  { id: 'gpt-4-turbo', name: 'GPT-4 Turbo (OpenAI)', provider: 'openai' },
  { id: 'claude-3-opus', name: 'Claude 3 Opus (Anthropic)', provider: 'anthropic' },
  { id: 'claude-3-sonnet', name: 'Claude 3 Sonnet (Anthropic)', provider: 'anthropic' },
  { id: 'gemini-pro', name: 'Gemini Pro (Google)', provider: 'google' },
  { id: 'llama-3-70b', name: 'Llama 3 70B (Meta)', provider: 'meta' },
];

interface TeamConfig {
  spymaster: string;
  operative: string;
}

const CodenamesLiveScreen: React.FC = () => {
  const navigate = useNavigate();
  const [redTeam, setRedTeam] = useState<TeamConfig>({
    spymaster: 'gpt-4o',
    operative: 'claude-3-opus',
  });
  const [blueTeam, setBlueTeam] = useState<TeamConfig>({
    spymaster: 'gemini-pro',
    operative: 'gpt-4-turbo',
  });
  const [isStarting, setIsStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleStartGame = async () => {
    setIsStarting(true);
    setError(null);

    try {
      const response = await fetch('/api/arena/codenames/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          players: [
            { id: 'red_spymaster', team: 'red', role: 'spymaster', type: 'llm', model: redTeam.spymaster },
            { id: 'red_operative', team: 'red', role: 'operative', type: 'llm', model: redTeam.operative },
            { id: 'blue_spymaster', team: 'blue', role: 'spymaster', type: 'llm', model: blueTeam.spymaster },
            { id: 'blue_operative', team: 'blue', role: 'operative', type: 'llm', model: blueTeam.operative },
          ],
          config: {
            word_list: 'standard',
            max_turns: 50,
          },
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to start game');
      }

      const data = await response.json();
      navigate(`/codenames/live/${data.game_id}`);
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

  const TeamConfigPanel = ({
    team,
    config,
    onChange,
  }: {
    team: 'red' | 'blue';
    config: TeamConfig;
    onChange: (config: TeamConfig) => void;
  }) => (
    <Paper
      sx={{
        p: 3,
        flex: 1,
        minWidth: 280,
        borderTop: `4px solid ${team === 'red' ? '#dc3545' : '#007bff'}`,
      }}
    >
      <Typography variant="h6" gutterBottom sx={{ color: team === 'red' ? '#dc3545' : '#007bff' }}>
        {team === 'red' ? 'Red Team' : 'Blue Team'}
      </Typography>

      <Box sx={{ mb: 2 }}>
        <Typography variant="subtitle2" gutterBottom>
          Spymaster
        </Typography>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
          Gives clues to help operatives find team words
        </Typography>
        <FormControl fullWidth size="small">
          <Select
            value={config.spymaster}
            onChange={(e) => onChange({ ...config, spymaster: e.target.value })}
          >
            {AVAILABLE_MODELS.map(model => (
              <MenuItem key={model.id} value={model.id}>
                {model.name}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Box>

      <Box>
        <Typography variant="subtitle2" gutterBottom>
          Operative
        </Typography>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
          Guesses words based on spymaster's clues
        </Typography>
        <FormControl fullWidth size="small">
          <Select
            value={config.operative}
            onChange={(e) => onChange({ ...config, operative: e.target.value })}
          >
            {AVAILABLE_MODELS.map(model => (
              <MenuItem key={model.id} value={model.id}>
                {model.name}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Box>
    </Paper>
  );

  return (
    <Box className="codenames-live-screen">
      {/* Header */}
      <Paper className="header" elevation={2}>
        <Button startIcon={<ArrowBackIcon />} onClick={handleBack} sx={{ color: 'white' }}>
          Back
        </Button>
        <Typography variant="h5">
          Codenames - LLM Arena
        </Typography>
        <Box sx={{ width: 100 }} />
      </Paper>

      <Container maxWidth="md" sx={{ py: 4 }}>
        {/* Info Alert */}
        <Alert severity="info" sx={{ mb: 3 }}>
          <Typography variant="body2">
            <strong>How it works:</strong> Configure LLM models for each team role below.
            The Spymaster sees all card colors and gives clues. The Operative only sees
            revealed cards and must guess based on the clue.
          </Typography>
        </Alert>

        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        {/* Team Configuration */}
        <Paper sx={{ p: 3, mb: 3, backgroundColor: '#2c3e50' }}>
          <Typography variant="h6" gutterBottom sx={{ color: 'white' }}>
            Configure Teams
          </Typography>
          <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.7)', mb: 3 }}>
            Select which LLM model plays each role. Different models have different strengths!
          </Typography>

          <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
            <TeamConfigPanel team="red" config={redTeam} onChange={setRedTeam} />
            <TeamConfigPanel team="blue" config={blueTeam} onChange={setBlueTeam} />
          </Box>
        </Paper>

        {/* Game Rules Summary */}
        <Paper sx={{ p: 3, mb: 3, backgroundColor: '#2c3e50' }}>
          <Typography variant="h6" gutterBottom sx={{ color: 'white' }}>
            Game Rules
          </Typography>
          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
            <Chip label="5x5 Word Grid" variant="outlined" sx={{ color: 'white', borderColor: 'rgba(255,255,255,0.3)' }} />
            <Chip label="9 Red, 8 Blue" variant="outlined" sx={{ color: 'white', borderColor: 'rgba(255,255,255,0.3)' }} />
            <Chip label="7 Bystanders" variant="outlined" sx={{ color: 'white', borderColor: 'rgba(255,255,255,0.3)' }} />
            <Chip label="1 Assassin" variant="outlined" sx={{ color: 'white', borderColor: 'rgba(255,255,255,0.3)' }} />
          </Box>
          <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.7)', mt: 2 }}>
            First team to find all their agents wins. Hitting the assassin = instant loss!
          </Typography>
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
            onClick={() => navigate('/codenames/replay')}
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

export default CodenamesLiveScreen;
