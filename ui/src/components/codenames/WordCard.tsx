import React from 'react';
import './WordCard.scss';

export type CardType = 'red' | 'blue' | 'bystander' | 'assassin' | 'hidden';

interface WordCardProps {
  word: string;
  cardType: CardType;
  isRevealed: boolean;
  onClick?: () => void;
  disabled?: boolean;
  showType?: boolean;  // Show card type overlay for spymaster view
}

const WordCard: React.FC<WordCardProps> = ({
  word,
  cardType,
  isRevealed,
  onClick,
  disabled = false,
  showType = false,
}) => {
  const getCardClass = () => {
    const classes = ['word-card'];

    if (isRevealed) {
      classes.push('revealed');
      classes.push(cardType);
    } else {
      classes.push('unrevealed');
      if (showType && cardType !== 'hidden') {
        classes.push(`hint-${cardType}`);
      }
    }

    if (disabled) {
      classes.push('disabled');
    }

    return classes.join(' ');
  };

  const getTypeIndicator = () => {
    if (!showType || cardType === 'hidden') return null;

    const indicators: Record<CardType, string> = {
      red: 'R',
      blue: 'B',
      bystander: '-',
      assassin: 'X',
      hidden: '',
    };

    return (
      <div className={`type-indicator ${cardType}`}>
        {indicators[cardType]}
      </div>
    );
  };

  return (
    <div
      className={getCardClass()}
      onClick={disabled ? undefined : onClick}
      role="button"
      tabIndex={disabled ? -1 : 0}
      onKeyPress={(e) => {
        if (!disabled && (e.key === 'Enter' || e.key === ' ')) {
          onClick?.();
        }
      }}
    >
      <span className="word-text">{word}</span>
      {getTypeIndicator()}

      {/* Assassin warning overlay */}
      {cardType === 'assassin' && showType && !isRevealed && (
        <div className="assassin-warning">!</div>
      )}
    </div>
  );
};

export default WordCard;
