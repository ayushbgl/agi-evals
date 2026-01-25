import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button, FormControl, InputLabel, Select, MenuItem, Box, Typography } from "@mui/material";

import "./HomePage.scss";

// Available games
type GameType = "catan" | "codenames";

export default function HomePage() {
  const [selectedGame, setSelectedGame] = useState<GameType>("catan");
  const navigate = useNavigate();

  const handleStartGame = () => {
    // Navigate to the live game view for the selected game
    navigate(`/${selectedGame}/play`);
  };

  const handleViewReplays = () => {
    navigate(`/${selectedGame}/replay`);
  };

  return (
    <div className="home-page">
      <h1 className="logo">Game Arena</h1>
      <Typography variant="subtitle1" sx={{ color: 'rgba(255,255,255,0.7)', mb: 3 }}>
        LLM Benchmarking Platform
      </Typography>

      {/* Game Selector */}
      <Box className="game-selector">
        <FormControl variant="outlined" size="small" className="game-select-control">
          <InputLabel id="game-select-label" sx={{ color: 'white' }}>Select Game</InputLabel>
          <Select
            labelId="game-select-label"
            value={selectedGame}
            onChange={(e) => setSelectedGame(e.target.value as GameType)}
            label="Select Game"
            sx={{
              color: 'white',
              '.MuiOutlinedInput-notchedOutline': { borderColor: 'rgba(255,255,255,0.5)' },
              '&:hover .MuiOutlinedInput-notchedOutline': { borderColor: 'white' },
              '.MuiSvgIcon-root': { color: 'white' },
            }}
          >
            <MenuItem value="catan">Settlers of Catan</MenuItem>
            <MenuItem value="codenames">Codenames</MenuItem>
          </Select>
        </FormControl>
      </Box>

      <div className="switchable">
        {selectedGame === "catan" ? (
          /* Catan Options */
          <>
            <Typography variant="h5" className="game-title">Settlers of Catan</Typography>
            <Box className="game-description">
              <Typography variant="body2" sx={{ opacity: 0.8, mb: 1 }}>
                Classic strategy board game where players collect resources,
                build settlements and cities, and trade to earn victory points.
              </Typography>
              <ul>
                <li>2-4 LLM players</li>
                <li>Resource management & trading</li>
                <li>First to 10 victory points wins</li>
              </ul>
            </Box>
          </>
        ) : (
          /* Codenames Options */
          <>
            <Typography variant="h5" className="game-title">Codenames</Typography>
            <Box className="game-description">
              <Typography variant="body2" sx={{ opacity: 0.8, mb: 1 }}>
                Word association game where Spymasters give clues to help
                their Operatives find their team's agents on the board.
              </Typography>
              <ul>
                <li>Two teams: Red vs Blue</li>
                <li>Spymasters give word clues</li>
                <li>Operatives guess words</li>
                <li>Avoid the Assassin!</li>
              </ul>
            </Box>
          </>
        )}

        <Box className="action-buttons">
          <Button
            variant="contained"
            color="primary"
            size="large"
            onClick={handleStartGame}
          >
            Start LLM Game
          </Button>
          <Button
            variant="outlined"
            sx={{ color: 'white', borderColor: 'rgba(255,255,255,0.5)' }}
            onClick={handleViewReplays}
          >
            View Replays
          </Button>
        </Box>
      </div>
    </div>
  );
}
