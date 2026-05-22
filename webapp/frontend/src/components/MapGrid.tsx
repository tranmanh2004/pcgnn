import type { Grid } from "../api";

interface Props {
  grid: Grid;
  /** Per-tile size in px. If omitted, scales down for larger grids to fit ~maxPx box. */
  size?: number;
  /** Target max box size in px when auto-sizing. */
  maxPx?: number;
}

const COLORS: Record<number, string> = {
  0: "#0f172a",  // WALL
  1: "#e2e8f0",  // FLOOR
  2: "#22c55e",  // PLAYER
  3: "#ef4444",  // ENEMY
};

export function MapGrid({ grid, size, maxPx = 196 }: Props) {
  const rows = grid.length;
  const cols = rows ? grid[0].length : 0;
  const tileSize = size ?? Math.max(3, Math.floor(maxPx / Math.max(rows, cols)));
  const width = cols * tileSize;
  const height = rows * tileSize;

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className="map-svg"
      shapeRendering="crispEdges"
    >
      {grid.flatMap((row, r) =>
        row.map((v, c) => (
          <rect
            key={`${r}-${c}`}
            x={c * tileSize}
            y={r * tileSize}
            width={tileSize}
            height={tileSize}
            fill={COLORS[v] ?? "#64748b"}
            stroke="#334155"
            strokeWidth={tileSize >= 6 ? 0.5 : 0}
          />
        ))
      )}
    </svg>
  );
}
