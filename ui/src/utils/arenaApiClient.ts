/**
 * Arena API Client - Load and validate Arena game logs
 */

import type { ArenaGameLog } from "./arenaApi.types";

/**
 * Load an Arena game log from a URL
 */
export async function loadArenaLog(url: string): Promise<ArenaGameLog> {
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Failed to load log: HTTP ${response.status}`);
  }

  const data = await response.json();

  // Basic validation
  if (!data.schema_version || !data.game_id || !data.turns) {
    throw new Error("Invalid Arena game log format");
  }

  return data as ArenaGameLog;
}

/**
 * Load an Arena game log from a File object
 */
export async function loadArenaLogFromFile(file: File): Promise<ArenaGameLog> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = (event) => {
      try {
        const data = JSON.parse(event.target?.result as string);

        // Basic validation
        if (!data.schema_version || !data.game_id || !data.turns) {
          throw new Error("Invalid Arena game log format");
        }

        resolve(data as ArenaGameLog);
      } catch (err) {
        reject(err);
      }
    };

    reader.onerror = () => reject(new Error("Failed to read file"));
    reader.readAsText(file);
  });
}

/**
 * List available Arena game logs from the server
 */
export async function listArenaLogs(
  baseUrl: string = "/api/arena/logs"
): Promise<{ game_id: string; created_at: string; players: string[] }[]> {
  const response = await fetch(baseUrl);

  if (!response.ok) {
    throw new Error(`Failed to list logs: HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Validate that a log matches the expected schema version
 */
export function validateLogVersion(log: ArenaGameLog): boolean {
  // Currently we support v1.x
  return log.schema_version.startsWith("1.");
}

/**
 * Get a summary of the game from the log
 */
export function getGameSummary(log: ArenaGameLog): {
  gameId: string;
  gameType: string;
  players: { id: string; type: string; model?: string }[];
  totalTurns: number;
  winner?: string;
  duration?: number;
} {
  return {
    gameId: log.game_id,
    gameType: log.game_type,
    players: log.players.map((p) => ({
      id: p.id,
      type: p.type,
      model: p.model_name,
    })),
    totalTurns: log.turns.length,
    winner: log.result?.winner_id,
    duration: log.duration_seconds,
  };
}
