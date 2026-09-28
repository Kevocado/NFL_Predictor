import type {
  GamePrediction,
  GameSummary,
  PlayerPropPrediction,
  RetrainResponse,
  TrackRecord,
} from "../types";

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "/api";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json();
}

/**
 * A successful response that is not quite the data on its own.
 *
 * The props route sets `X-Player-Props-Stale` when the rows it returns were
 * carried forward from an earlier snapshot build because this week's rebuild
 * failed. Those are still pregame projections for the right week -- which is all
 * the accuracy rule requires -- but a reader must be able to tell they are not
 * from the current build, or "stale" is a key nothing reads, which is the
 * defect. The header is additive, so the body is still a bare JSON array and
 * every existing reader of it is unaffected.
 */
export interface WithMeta<T> {
  data: T;
  stale: boolean;
}

async function getWithMeta<T>(path: string): Promise<WithMeta<T>> {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return { data: await res.json(), stale: res.headers.get("X-Player-Props-Stale") === "true" };
}

async function post<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, { method: "POST" });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const api = {
  games: (season: number, week: number) => get<GameSummary[]>(`/games?season=${season}&week=${week}`),
  gamePrediction: (season: number, week: number, gameId: string) =>
    get<GamePrediction>(`/games/${season}/${week}/${gameId}/prediction`),
  playerProps: (season: number, week: number) =>
    getWithMeta<PlayerPropPrediction[]>(`/players/${season}/${week}/props`),
  trackRecord: () => get<TrackRecord>("/track-record"),
  retrain: () => post<RetrainResponse>("/retrain"),
};
