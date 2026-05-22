import { Table, Text } from "@mantine/core";
import type { Summary } from "../api";

interface Props {
  rows: { label: string; summary: Summary; accent?: string }[];
}

const FIELDS: { key: keyof Summary; label: string; fmt: (v: number) => string }[] = [
  { key: "count", label: "Số map", fmt: (v) => String(v) },
  { key: "solvability", label: "Solvability", fmt: (v) => `${(v * 100).toFixed(1)}%` },
  { key: "wall_ratio", label: "Wall ratio", fmt: (v) => v.toFixed(3) },
  { key: "interior_wall_density", label: "Interior wall", fmt: (v) => v.toFixed(3) },
  { key: "reachable_ratio", label: "Reachable", fmt: (v) => v.toFixed(3) },
  { key: "shortest_path_length", label: "Path length", fmt: (v) => v.toFixed(1) },
  { key: "path_norm", label: "Path norm", fmt: (v) => v.toFixed(3) },
  { key: "dead_end_ratio", label: "Dead ends", fmt: (v) => v.toFixed(3) },
  { key: "branching_ratio", label: "Branching", fmt: (v) => v.toFixed(3) },
  { key: "leniency", label: "Leniency", fmt: (v) => v.toFixed(3) },
  { key: "astar_difficulty", label: "A* difficulty", fmt: (v) => v.toFixed(3) },
  { key: "difficulty_score", label: "Difficulty score", fmt: (v) => v.toFixed(3) },
];

export function MetricsTable({ rows }: Props) {
  return (
    <Table striped highlightOnHover withTableBorder withColumnBorders verticalSpacing="xs">
      <Table.Thead>
        <Table.Tr>
          <Table.Th>Chỉ số</Table.Th>
          {rows.map((r, i) => (
            <Table.Th key={i}>
              <Text c={r.accent} fw={600}>
                {r.label}
              </Text>
            </Table.Th>
          ))}
        </Table.Tr>
      </Table.Thead>
      <Table.Tbody>
        {FIELDS.map(({ key, label, fmt }) => (
          <Table.Tr key={key as string}>
            <Table.Td>{label}</Table.Td>
            {rows.map((r, i) => {
              const value = r.summary[key];
              return (
                <Table.Td key={i} className="metric-cell">
                  {typeof value === "number" ? fmt(value) : "-"}
                </Table.Td>
              );
            })}
          </Table.Tr>
        ))}
      </Table.Tbody>
    </Table>
  );
}
