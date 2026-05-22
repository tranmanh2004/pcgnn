import type { Grid } from "../api";

interface Props {
  grid: Grid;
  size?: number;
  showLegend?: boolean;
}

const COLORS: Record<number, string> = {
  0: "#0f172a",  // WALL
  1: "#f1f5f9",  // FLOOR
  2: "#22c55e",  // PLAYER
  3: "#ef4444",  // ENEMY
};

export function MapGrid({ grid, size = 14, showLegend = false }: Props) {
  const rows = grid.length;
  const cols = rows ? grid[0].length : 0;
  const width = cols * size;
  const height = rows * size;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        style={{ background: "#1e293b", borderRadius: 4 }}
        shapeRendering="crispEdges"
      >
        {grid.flatMap((row, r) =>
          row.map((v, c) => (
            <rect
              key={`${r}-${c}`}
              x={c * size}
              y={r * size}
              width={size}
              height={size}
              fill={COLORS[v] ?? "#64748b"}
              stroke="#334155"
              strokeWidth={0.5}
            />
          ))
        )}
      </svg>
      {showLegend && (
        <div style={{ display: "flex", gap: 10, fontSize: 11, color: "#94a3b8" }}>
          <Legend color={COLORS[0]} label="Wall" />
          <Legend color={COLORS[1]} label="Floor" />
          <Legend color={COLORS[2]} label="Player" />
          <Legend color={COLORS[3]} label="Enemy" />
        </div>
      )}
    </div>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
      <span style={{ width: 10, height: 10, background: color, border: "1px solid #334155", borderRadius: 2 }} />
      {label}
    </span>
  );
}
