/**
 * TypeScript types for Catan-Arena game logs.
 *
 * These types match the JSON schema defined in
 * catan_arena/storage/schemas/game_log_v1.json
 */

export type Color = "RED" | "BLUE" | "ORANGE" | "WHITE";
export type ResourceCard = "WOOD" | "BRICK" | "SHEEP" | "WHEAT" | "ORE";
export type DevelopmentCard =
  | "KNIGHT"
  | "VICTORY_POINT"
  | "ROAD_BUILDING"
  | "YEAR_OF_PLENTY"
  | "MONOPOLY";

export type PlayerType = "llm" | "human" | "random" | "minimax" | "mcts" | "value";

export type ActionType =
  | "ROLL"
  | "END_TURN"
  | "DISCARD"
  | "BUILD_ROAD"
  | "BUILD_SETTLEMENT"
  | "BUILD_CITY"
  | "BUY_DEVELOPMENT_CARD"
  | "PLAY_KNIGHT_CARD"
  | "PLAY_YEAR_OF_PLENTY"
  | "PLAY_MONOPOLY"
  | "PLAY_ROAD_BUILDING"
  | "MOVE_ROBBER"
  | "MARITIME_TRADE"
  | "OFFER_TRADE"
  | "ACCEPT_TRADE"
  | "REJECT_TRADE"
  | "CONFIRM_TRADE"
  | "CANCEL_TRADE";

export interface ArenaPlayer {
  id: string;
  color: Color;
  type: PlayerType;
  model_name?: string;
  llm_config?: {
    temperature?: number;
    max_tokens?: number;
  };
}

export interface ArenaGameConfig {
  map_type: "BASE" | "MINI";
  vps_to_win: number;
  max_turns: number;
  seed?: number;
}

export interface TileData {
  coordinate: [number, number, number];
  type: string;
  resource?: ResourceCard;
  number?: number;
}

export interface BuildingData {
  node_id: number;
  color: Color;
  type: "SETTLEMENT" | "CITY";
}

export interface RoadData {
  edge: [number, number];
  color: Color;
}

export interface BoardState {
  tiles: TileData[];
  buildings: BuildingData[];
  roads: RoadData[];
  robber_coordinate: [number, number, number];
}

export interface PlayerSummary {
  color: Color;
  victory_points_visible: number;
  resource_count: number;
  dev_card_count: number;
  knights_played: number;
  longest_road_length: number;
  has_longest_road: boolean;
  has_largest_army: boolean;
  settlements_left?: number;
  cities_left?: number;
  roads_left?: number;
}

export interface PublicTurnState {
  turn_number: number;
  current_player: number;
  current_prompt?: string;
  is_initial_build_phase: boolean;
  robber_coordinate: [number, number, number];
  bank_resources: number[];
  bank_dev_cards: number;
  longest_road_owner?: string;
  largest_army_owner?: string;
  player_summaries: Record<string, PlayerSummary>;
  board: BoardState;
}

export interface PrivateTurnState {
  hand_resources: Record<ResourceCard, number>;
  hand_dev_cards: Record<DevelopmentCard, number>;
  actual_victory_points: number;
  can_play_dev_card?: boolean;
  has_rolled?: boolean;
}

export interface GameAction {
  action_type: ActionType;
  value?: unknown;
  rationale?: string;
}

export interface LLMDecision {
  model_name: string;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  latency_ms: number;
  reasoning: string;
  raw_output: string;
  parsed_action?: GameAction;
  parsed_successfully: boolean;
  parse_error?: string;
}

export interface TurnRecord {
  turn_number: number;
  player_id: string;
  phase: string;
  public_state: PublicTurnState;
  private_states: Record<string, PrivateTurnState>;
  valid_actions: GameAction[];
  llm_decision?: LLMDecision;
  action: GameAction;
  action_result?: Record<string, unknown>;
  timestamp: string;
}

export interface GameStatistics {
  total_llm_calls: number;
  total_tokens_used: number;
  average_decision_time_ms: number;
  invalid_action_attempts?: Record<string, number>;
  actions_by_type?: Record<string, number>;
  actions_by_player?: Record<string, number>;
}

export interface GameResult {
  termination_reason: "victory" | "turn_limit" | "error";
  winner_id?: string;
  final_scores: Record<string, number>;
  total_turns: number;
  statistics: GameStatistics;
}

export interface ArenaGameLog {
  schema_version: string;
  game_id: string;
  game_type: string;
  created_at: string;
  duration_seconds?: number;
  players: ArenaPlayer[];
  config: ArenaGameConfig;
  initial_state?: BoardState;
  turns: TurnRecord[];
  result?: GameResult;
}

// View modes for fog of war
export type ViewModeType = "omniscient" | "player_pov" | "spectator";

export interface ViewMode {
  type: ViewModeType;
  player_id?: string; // For player_pov mode
}

// Computed visible state based on view mode
export interface VisibleGameState {
  board: BoardState;
  playerStates: Record<
    string,
    PlayerSummary & {
      handVisible: boolean;
      hand_resources?: Record<ResourceCard, number>;
      hand_dev_cards?: Record<DevelopmentCard, number>;
      actual_victory_points?: number;
    }
  >;
  showAllReasoning: boolean;
  showReasoningFor?: string;
}

// ============================================
// Codenames Types
// ============================================

export type CodenamesTeam = "red" | "blue";
export type CodenamesRole = "spymaster" | "operative";
export type CodenamesCardType = "red" | "blue" | "bystander" | "assassin";
export type CodenamesPhase = "spymaster_clue" | "operative_guess" | "game_over";

export interface CodenamesPlayer {
  id: string;
  team: CodenamesTeam;
  role: CodenamesRole;
  type: PlayerType;
  model_name?: string;
}

export interface CodenamesWordCell {
  word: string;
  revealed: boolean;
  card_type?: CodenamesCardType;
}

export interface CodenamesClue {
  word: string;
  number: number;
}

export interface CodenamesClueRecord {
  team: CodenamesTeam;
  word: string;
  number: number;
  turn: number;
}

export interface CodenamesGuessRecord {
  team: CodenamesTeam;
  player: string;
  word: string;
  card_type: CodenamesCardType;
  correct: boolean;
  turn: number;
}

export interface CodenamesPublicState {
  grid: CodenamesWordCell[][];
  revealed_words: string[];
  current_team: CodenamesTeam;
  current_phase: CodenamesPhase;
  current_clue: CodenamesClue | null;
  guesses_remaining: number;
  red_remaining: number;
  blue_remaining: number;
  turn_number: number;
  clue_history: CodenamesClueRecord[];
  guess_history: CodenamesGuessRecord[];
  game_over: boolean;
  winner: CodenamesTeam | null;
}

export interface CodenamesPrivateState {
  player_id: string;
  team: CodenamesTeam;
  role: CodenamesRole;
  card_types?: Record<string, CodenamesCardType>; // Only for spymasters
  key_card?: string[][]; // Visual key card for spymasters
}

export interface CodenamesAction {
  action_type: "GIVE_CLUE" | "GUESS" | "PASS";
  clue?: string;
  number?: number;
  word?: string;
  reasoning?: string;
}

export interface CodenamesTurnRecord {
  turn_number: number;
  player_id: string;
  role: CodenamesRole;
  phase: CodenamesPhase;
  public_state: CodenamesPublicState;
  private_states: Record<string, CodenamesPrivateState>;
  valid_actions: CodenamesAction[];
  llm_decision?: LLMDecision;
  action: CodenamesAction;
  action_result?: {
    card_type?: CodenamesCardType;
    correct?: boolean;
    turn_continues?: boolean;
    game_over?: boolean;
    winner?: CodenamesTeam;
  };
  timestamp: string;
}

export interface CodenamesGameConfig {
  word_list: "standard" | "easy" | "tech" | "combined";
  max_turns: number;
  starting_team?: CodenamesTeam;
}

export interface CodenamesGameResult {
  termination_reason: "victory" | "assassin" | "turn_limit" | "error";
  winner?: CodenamesTeam;
  final_scores: {
    red: { found: number; total: number; remaining: number };
    blue: { found: number; total: number; remaining: number };
  };
  total_turns: number;
  loss_reason?: string;
}

export interface CodenamesGameLog {
  schema_version: string;
  game_id: string;
  game_type: "codenames";
  created_at: string;
  duration_seconds?: number;
  players: CodenamesPlayer[];
  config: CodenamesGameConfig;
  initial_state?: {
    grid: string[][];
    card_types: Record<string, CodenamesCardType>;
  };
  turns: CodenamesTurnRecord[];
  result?: CodenamesGameResult;
}

// Union type for all game logs
export type GameLog = ArenaGameLog | CodenamesGameLog;

// Type guard functions
export function isCatanGameLog(log: GameLog): log is ArenaGameLog {
  return log.game_type === "catan";
}

export function isCodenamesGameLog(log: GameLog): log is CodenamesGameLog {
  return log.game_type === "codenames";
}
