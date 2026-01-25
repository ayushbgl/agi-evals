import React from 'react';
import WordCard from './WordCard';
import TurnIndicator from './TurnIndicator';
import './CodenamesBoard.scss';

export type CardType = 'red' | 'blue' | 'bystander' | 'assassin' | 'hidden';

export interface WordCellState {
  word: string;
  revealed: boolean;
  card_type?: string;
}

export interface CodenamesBoardProps {
  grid: WordCellState[][];
  cardTypes?: Record<string, string>;  // Only visible in spymaster view
  currentClue?: { word: string; number: number } | null;
  currentTeam: 'red' | 'blue';
  currentRole: 'spymaster' | 'operative';
  viewMode: 'spymaster' | 'operative' | 'spectator';
  redRemaining: number;
  blueRemaining: number;
  onCardClick?: (word: string) => void;
  gameOver?: boolean;
  winner?: string | null;
}

const CodenamesBoard: React.FC<CodenamesBoardProps> = ({
  grid,
  cardTypes,
  currentClue,
  currentTeam,
  currentRole,
  viewMode,
  redRemaining,
  blueRemaining,
  onCardClick,
  gameOver = false,
  winner = null,
}) => {
  const getCardType = (word: string, revealed: boolean, cellCardType?: string): CardType => {
    // If revealed, show the actual card type
    if (revealed && cellCardType) {
      return cellCardType as CardType;
    }

    // In spymaster view or spectator, show all card types
    if ((viewMode === 'spymaster' || viewMode === 'spectator') && cardTypes) {
      return (cardTypes[word] || 'hidden') as CardType;
    }

    // Otherwise, hide the card type
    return 'hidden';
  };

  return (
    <div className="codenames-board">
      {/* Header with turn indicator and scores */}
      <div className="board-header">
        <div className="score-display red">
          <span className="team-label">RED</span>
          <span className="score">{redRemaining}</span>
        </div>

        <TurnIndicator
          currentTeam={currentTeam}
          currentRole={currentRole}
          gameOver={gameOver}
          winner={winner}
        />

        <div className="score-display blue">
          <span className="team-label">BLUE</span>
          <span className="score">{blueRemaining}</span>
        </div>
      </div>

      {/* Current clue display */}
      {currentClue && !gameOver && (
        <div className={`current-clue ${currentTeam}`}>
          <span className="clue-label">CLUE:</span>
          <span className="clue-word">{currentClue.word}</span>
          <span className="clue-number">{currentClue.number}</span>
        </div>
      )}

      {/* 5x5 Grid */}
      <div className="grid-container">
        {grid.map((row, rowIndex) => (
          <div key={rowIndex} className="grid-row">
            {row.map((cell, colIndex) => (
              <WordCard
                key={`${rowIndex}-${colIndex}`}
                word={cell.word}
                cardType={getCardType(cell.word, cell.revealed, cell.card_type)}
                isRevealed={cell.revealed}
                onClick={() => onCardClick?.(cell.word)}
                disabled={cell.revealed || gameOver}
                showType={viewMode === 'spymaster' || viewMode === 'spectator'}
              />
            ))}
          </div>
        ))}
      </div>

      {/* View mode indicator */}
      <div className="view-mode-indicator">
        Viewing as: <span className="mode">{viewMode}</span>
      </div>
    </div>
  );
};

export default CodenamesBoard;
