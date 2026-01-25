import React from 'react';
import './TurnIndicator.scss';

interface TurnIndicatorProps {
  currentTeam: 'red' | 'blue';
  currentRole: 'spymaster' | 'operative';
  playerName?: string;
  gameOver?: boolean;
  winner?: string | null;
}

const TurnIndicator: React.FC<TurnIndicatorProps> = ({
  currentTeam,
  currentRole,
  playerName,
  gameOver = false,
  winner = null,
}) => {
  if (gameOver) {
    return (
      <div className={`turn-indicator game-over ${winner}`}>
        <span className="game-over-text">GAME OVER</span>
        {winner && (
          <span className="winner-text">{winner.toUpperCase()} WINS!</span>
        )}
      </div>
    );
  }

  return (
    <div className={`turn-indicator ${currentTeam}`}>
      <div className="team-badge">{currentTeam.toUpperCase()}</div>
      <div className="role-text">{currentRole}</div>
      {playerName && <div className="player-name">({playerName})</div>}
    </div>
  );
};

export default TurnIndicator;
