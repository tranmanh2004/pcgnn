export type Grid = number[][];

export interface MapMetrics {
  has_player: boolean;
  has_enemy: boolean;
  solvable: boolean;
  wall_ratio: number;
  interior_wall_density: number;
  walkable_cells: number;
  reachable_ratio: number;
  shortest_path_length: number;
  path_norm: number;
  dead_end_ratio: number;
  branching_ratio: number;
  leniency: number;
  astar_difficulty: number;
  difficulty_score: number;
}

export interface MapPayload {
  index: number;
  grid: Grid;
  metrics: MapMetrics;
}

export interface ClassifiedMap extends MapPayload {
  score_tier: string;
  range_tier: string;
  percentile_tier: string;
}

export interface Summary {
  count: number;
  solvability: number;
  wall_ratio: number;
  interior_wall_density: number;
  reachable_ratio: number;
  shortest_path_length: number;
  path_norm: number;
  dead_end_ratio: number;
  branching_ratio: number;
  leniency: number;
  astar_difficulty: number;
  difficulty_score: number;
}

export type ModelName = "baseline" | "improved";

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${path} ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export interface GenerateResponse {
  model: ModelName;
  count: number;
  seed: number;
  height: number;
  width: number;
  maps: MapPayload[];
  summary: Summary;
}

export function generate(req: {
  model: ModelName;
  count: number;
  seed: number;
  width?: number;
  height?: number;
  perturb?: boolean;
}): Promise<GenerateResponse> {
  return postJson("/api/generate", req);
}

export interface CompareResponse {
  count: number;
  seed: number;
  baseline: MapPayload[];
  improved: MapPayload[];
  summary: { baseline: Summary; improved: Summary };
}

export function compare(req: {
  count: number;
  seed: number;
  width?: number;
  height?: number;
  perturb?: boolean;
}): Promise<CompareResponse> {
  return postJson("/api/compare", req);
}

export interface ClassifyResponse {
  model: ModelName;
  count: number;
  seed: number;
  easy_ratio: number;
  medium_ratio: number;
  maps: ClassifiedMap[];
  distribution: {
    score_tier: Record<string, number>;
    range_tier: Record<string, number>;
    percentile_tier: Record<string, number>;
  };
}

export function classify(req: {
  model: ModelName;
  count: number;
  seed: number;
  width?: number;
  height?: number;
  perturb?: boolean;
  easy_ratio: number;
  medium_ratio: number;
}): Promise<ClassifyResponse> {
  return postJson("/api/classify", req);
}
