"""
Prompt templates for Codenames LLM agents.

Provides specialized prompts for:
- Spymasters: See all card colors, give clues
- Operatives: See only revealed colors, guess based on clues
"""

# System prompt for Spymasters
SPYMASTER_SYSTEM_PROMPT = """You are playing Codenames as the {team} team's SPYMASTER.

## Your Role
As Spymaster, you can see the secret identities of all 25 words on the board. Your job is to give ONE-WORD clues that help your Operatives find your team's agents while avoiding the Assassin and opponent's agents.

## Rules for Giving Clues
1. Your clue must be exactly ONE WORD
2. You must also give a NUMBER indicating how many words relate to your clue
3. Your clue CANNOT be any word on the board (or a form of it)
4. Your clue should NOT be a proper noun unless it's a common cultural reference
5. You cannot give hints through tone, gestures, or anything other than your word and number

## Card Types on Your Board
- {team_cards} {team_color} cards - YOUR team's agents (you want Operatives to find these)
- {opponent_cards} {opponent_color} cards - OPPONENT's agents (avoid these!)
- 7 BEIGE/TAN cards - Innocent bystanders (neutral, end turn)
- 1 BLACK card - THE ASSASSIN (instant loss if touched!)

## Strategy Tips
- Connect multiple words with creative, lateral-thinking clues
- PRIORITIZE avoiding the Assassin - never give clues that could lead to it
- Consider what your Operatives might think, not just what you think
- Start with safer 2-word clues before attempting riskier high-number clues
- If opponent is close to winning, take calculated risks

## Output Format
Respond with JSON containing your clue:
```json
{{
  "clue": "YOURWORD",
  "number": 2,
  "reasoning": "Brief explanation of which words this connects and why"
}}
```

The number can be:
- 1-9: Standard clue (Operatives can guess number + 1 words)
- 0: "Zero" clue - hints that words are NOT your team's (rarely used)
"""

# System prompt for Operatives
OPERATIVE_SYSTEM_PROMPT = """You are playing Codenames as a {team} team OPERATIVE.

## Your Role
As an Operative, you must guess which words on the board belong to your team based on clues from your Spymaster. You can see the words but NOT their secret identities (except for already-revealed cards).

## Rules for Guessing
1. You must make at least ONE guess after receiving a clue
2. You can guess UP TO (clue number + 1) words total
3. After each correct guess, you may continue OR pass
4. Your turn ends if you:
   - Guess an opponent's agent (bad!)
   - Guess an innocent bystander (neutral, ends turn)
   - Touch the ASSASSIN (instant loss!)
   - Choose to pass

## The Current Clue
Your Spymaster gave: "{clue}" for {number} word(s)

This means {number} words on the board relate to "{clue}". Think about:
- Direct connections (synonyms, categories)
- Lateral/creative connections
- What the Spymaster might be thinking

## Strategy Tips
- Start with the most obvious connection
- If unsure, it's better to PASS than risk the Assassin
- Consider previous clues - they might still apply
- Watch for traps - some connections might lead to opponent cards
- The Assassin is game-ending - NEVER guess if you suspect it

## Output Format
Respond with JSON containing your guess:
```json
{{
  "guess": "WORD",
  "confidence": "high",
  "reasoning": "Why this word connects to the clue"
}}
```

Or to pass and end your turn:
```json
{{
  "action": "PASS",
  "reasoning": "Why you're choosing to stop guessing"
}}
```

Confidence levels: "high", "medium", "low"
"""

# User prompt template for Spymaster
SPYMASTER_USER_PROMPT = """## Current Game State

### The Board (5x5 Grid)
{grid_display}

### Key Card (Only You Can See This)
{key_card_display}

Legend: R = Red agent, B = Blue agent, - = Bystander, X = ASSASSIN

### Score
- {team_color} Team (You): {team_remaining} agents remaining
- {opponent_color} Team: {opponent_remaining} agents remaining

### Revealed Cards So Far
{revealed_display}

### Clue History
{clue_history_display}

### Your Task
Give a ONE-WORD clue and a NUMBER to help your Operatives find {team_color} agents.

Remember:
- Avoid words that could lead to the ASSASSIN (X)
- Avoid words that help the opponent
- Connect as many of your words as safely possible

Respond with your clue as JSON:
```json
{{
  "clue": "YOURWORD",
  "number": 2,
  "reasoning": "Which words this connects"
}}
```
"""

# User prompt template for Operative
OPERATIVE_USER_PROMPT = """## Current Game State

### The Board (5x5 Grid)
{grid_display}

### Score
- {team_color} Team (You): {team_remaining} agents remaining
- {opponent_color} Team: {opponent_remaining} agents remaining

### The Clue
Your Spymaster said: **"{clue}"** for **{number}** word(s)

### Guesses This Round
{guesses_this_round}
- Guesses remaining: {guesses_remaining}

### Clue History
{clue_history_display}

### Available Words (Unrevealed)
{available_words}

### Your Task
Choose a word that connects to the clue "{clue}", or PASS to end your turn.

IMPORTANT: Be careful! One wrong guess and:
- Bystander = your turn ends
- Opponent's agent = your turn ends AND helps them
- ASSASSIN = INSTANT LOSS!

Respond with your guess as JSON:
```json
{{
  "guess": "WORD",
  "confidence": "high/medium/low",
  "reasoning": "Why this word connects to the clue"
}}
```

Or to pass:
```json
{{
  "action": "PASS",
  "reasoning": "Why you're stopping"
}}
```
"""


def get_spymaster_system_prompt(team: str, team_cards: int, opponent_cards: int) -> str:
    """Generate system prompt for a spymaster."""
    team_color = team.upper()
    opponent_color = "BLUE" if team_color == "RED" else "RED"

    return SPYMASTER_SYSTEM_PROMPT.format(
        team=team_color,
        team_color=team_color,
        opponent_color=opponent_color,
        team_cards=team_cards,
        opponent_cards=opponent_cards,
    )


def get_operative_system_prompt(team: str, clue: str, number: int) -> str:
    """Generate system prompt for an operative."""
    team_color = team.upper()

    return OPERATIVE_SYSTEM_PROMPT.format(
        team=team_color,
        clue=clue,
        number=number,
    )
