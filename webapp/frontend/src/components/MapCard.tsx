import type { MapMetrics } from "../api";
import { MapGrid } from "./MapGrid";

interface Props {
  index: number;
  grid: number[][];
  metrics: MapMetrics;
  tier?: string;
  size?: number;
}

export function MapCard({ index, grid, metrics, tier, size = 14 }: Props) {
  return (
    <div className="map-card">
      <div className="header-row">
        <strong>#{String(index).padStart(3, "0")}</strong>
        {tier && <span className={`tier-badge ${tier}`}>{tier}</span>}
      </div>
      <MapGrid grid={grid} size={size} />
      <div className="meta">
        <div>solv: {metrics.solvable ? "✓" : "✗"}</div>
        <div>wall: {metrics.wall_ratio.toFixed(2)}</div>
        <div>path: {metrics.shortest_path_length}</div>
        <div>diff: {metrics.difficulty_score.toFixed(3)}</div>
      </div>
    </div>
  );
}
