import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { SnackbarProvider } from "notistack";
import { createTheme, ThemeProvider } from "@mui/material/styles";
import { blue, green } from "@mui/material/colors";
import Fade from "@mui/material/Fade";

import GameScreen from "./pages/GameScreen";
import HomePage from "./pages/HomePage";
import { StateProvider } from "./store";

import "./App.scss";
import ReplayScreen from "./pages/ReplayScreen";
import ArenaReplayScreen from "./pages/ArenaReplayScreen";
import CodenamesReplayScreen from "./pages/CodenamesReplayScreen";
import CatanLiveScreen from "./pages/CatanLiveScreen";
import CodenamesLiveScreen from "./pages/CodenamesLiveScreen";
import CatanGameScreen from "./pages/CatanGameScreen";
import CodenamesGameScreen from "./pages/CodenamesGameScreen";

const theme = createTheme({
  palette: {
    primary: {
      main: blue[900],
    },
    secondary: {
      main: green[900],
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <StateProvider>
        <SnackbarProvider
          classes={{ containerRoot: "snackbar-container" }}
          maxSnack={1}
          autoHideDuration={1000}
          TransitionComponent={Fade}
          TransitionProps={{ timeout: 100 }}
        >
          <Router>
            <Routes>
              {/* Game configuration screens */}
              <Route path="/catan/play" element={<CatanLiveScreen />} />
              <Route path="/codenames/play" element={<CodenamesLiveScreen />} />

              {/* Live game screens (with game ID) */}
              <Route path="/catan/live/:gameId" element={<CatanGameScreen />} />
              <Route path="/codenames/live/:gameId" element={<CodenamesGameScreen />} />

              {/* Replay screens */}
              <Route path="/catan/replay/:gameId?" element={<ArenaReplayScreen />} />
              <Route path="/catan/replay" element={<ArenaReplayScreen />} />
              <Route path="/codenames/replay/:gameId?" element={<CodenamesReplayScreen />} />
              <Route path="/codenames/replay" element={<CodenamesReplayScreen />} />

              {/* Legacy routes for existing game engine */}
              <Route
                path="/games/:gameId/states/:stateIndex"
                element={<GameScreen replayMode={true} />}
              />
              <Route path="/replays/:gameId" element={<ReplayScreen />} />
              <Route
                path="/games/:gameId"
                element={<GameScreen replayMode={false} />}
              />

              {/* Home */}
              <Route path="/" element={<HomePage />} />
            </Routes>
          </Router>
        </SnackbarProvider>
      </StateProvider>
    </ThemeProvider>
  );
}

export default App;
