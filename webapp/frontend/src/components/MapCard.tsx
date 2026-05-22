import { Card, Group, Badge, Stack, Text } from "@mantine/core";
import { IconCheck, IconX } from "@tabler/icons-react";
import type { MapMetrics } from "../api";
import { MapGrid } from "./MapGrid";

interface Props {
  index: number;
  grid: number[][];
  metrics: MapMetrics;
  tier?: string;
  size?: number;
  maxPx?: number;
}

const TIER_COLORS: Record<string, string> = {
  easy: "green",
  medium: "yellow",
  hard: "red",
  unclassified: "gray",
};

export function MapCard({ index, grid, metrics, tier, size, maxPx = 196 }: Props) {
  return (
    <Card withBorder padding="sm" radius="md" className="map-card-clickable">
      <Stack gap={8}>
        <Group justify="space-between" gap="xs">
          <Text size="sm" fw={600} ff="monospace">
            #{String(index).padStart(3, "0")}
          </Text>
          <Group gap={4}>
            {tier && (
              <Badge color={TIER_COLORS[tier] ?? "gray"} variant="light" size="sm">
                {tier}
              </Badge>
            )}
            <Badge
              color={metrics.solvable ? "teal" : "red"}
              variant="light"
              size="sm"
              leftSection={
                metrics.solvable ? <IconCheck size={10} /> : <IconX size={10} />
              }
            >
              {metrics.solvable ? "solv" : "fail"}
            </Badge>
          </Group>
        </Group>
        <Group justify="center">
          <MapGrid grid={grid} size={size} maxPx={maxPx} />
        </Group>
        <Group justify="space-between" gap="xs">
          <Text size="xs" c="dimmed" className="metric-cell">
            wall {metrics.wall_ratio.toFixed(2)}
          </Text>
          <Text size="xs" c="dimmed" className="metric-cell">
            path {metrics.shortest_path_length}
          </Text>
          <Text size="xs" c="dimmed" className="metric-cell">
            diff {metrics.difficulty_score.toFixed(2)}
          </Text>
        </Group>
      </Stack>
    </Card>
  );
}
