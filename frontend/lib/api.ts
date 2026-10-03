export const API = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export type TeamStats = {
  team_id: number;
  players_tracked: number;
  total_distance_m: number;
  avg_x: number;
  avg_y: number;
};

export type Summary = {
  fps: number;
  frames: number;
  first_frame: number;
  last_frame: number;
  clip_seconds: number;
  tracks: number;
  team_names: string[];
  pitch: { length_m: number; width_m: number };
  bounds: { x_min: number; x_max: number; y_min: number; y_max: number };
  teams: Record<string, TeamStats>;
};

export type TrackRow = {
  frame: number;
  track_id: number;
  team: string;
  role: string;
  pitch_x: number;
  pitch_y: number;
};

export type PlayerStat = {
  track_id: number;
  team: string;
  role: string;
  distance_m: number;
  avg_speed_ms: number;
  seen_s: number;
};

async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${API}${path}`);
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return r.json() as Promise<T>;
}

export const getSummary = () => get<Summary>("/api/summary");
export const getPlayers = () => get<PlayerStat[]>("/api/players");

/** The API caps a range at 300 frames, so fetch the whole clip in chunks. */
export async function getAllTracks(first: number, last: number): Promise<TrackRow[]> {
  const jobs: Promise<TrackRow[]>[] = [];
  for (let s = first; s <= last; s += 300) {
    jobs.push(get<TrackRow[]>(`/api/tracks?start=${s}&end=${Math.min(s + 299, last)}`));
  }
  return (await Promise.all(jobs)).flat();
}

export const heatmapUrl = (team: string) => `${API}/api/heatmap/${team}`;
export const avgPositionsUrl = `${API}/api/avg-positions`;
export const reportUrl = `${API}/api/report`;

export const TEAM_COLOR: Record<string, string> = {
  red: "#ef4444",
  green: "#4ade80",
};
export const teamColor = (t: string) => TEAM_COLOR[t] ?? "#60a5fa";
