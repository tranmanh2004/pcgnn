import { useMemo, useState } from "react";
import {
  Alert,
  Button,
  Card,
  Center,
  Group,
  Loader,
  NumberInput,
  SimpleGrid,
  Stack,
  Text,
  Title,
} from "@mantine/core";
import { BarChart } from "@mantine/charts";
import {
  IconAlertCircle,
  IconArrowsLeftRight,
  IconChartBar,
} from "@tabler/icons-react";
import { compare, type CompareResponse } from "../api";
import { MapCard } from "../components/MapCard";
import { MetricsTable } from "../components/MetricsTable";

const CHART_METRICS: { key: keyof CompareResponse["summary"]["baseline"]; label: string }[] = [
  { key: "solvability", label: "Solvability" },
  { key: "wall_ratio", label: "Wall" },
  { key: "reachable_ratio", label: "Reachable" },
  { key: "path_norm", label: "Path norm" },
  { key: "dead_end_ratio", label: "Dead ends" },
  { key: "branching_ratio", label: "Branching" },
  { key: "leniency", label: "Leniency" },
  { key: "astar_difficulty", label: "A* diff" },
  { key: "difficulty_score", label: "Difficulty" },
];

export function ComparePage() {
  const [count, setCount] = useState<number>(8);
  const [seed, setSeed] = useState<number>(0);
  const [width, setWidth] = useState<number>(14);
  const [height, setHeight] = useState<number>(14);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<CompareResponse | null>(null);

  async function onSubmit() {
    setLoading(true);
    setError(null);
    try {
      const result = await compare({ count, seed, width, height });
      setData(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  const chartData = useMemo(() => {
    if (!data) return [];
    return CHART_METRICS.map(({ key, label }) => ({
      metric: label,
      baseline: Number((data.summary.baseline[key] as number).toFixed(3)),
      improved: Number((data.summary.improved[key] as number).toFixed(3)),
    }));
  }, [data]);

  return (
    <Stack gap="md">
      <Stack gap={2}>
        <Title order={3}>So sánh baseline vs improved</Title>
        <Text c="dimmed" size="sm">
          Sinh song song N map từ mỗi model với cùng seed. So sánh chỉ số trung bình + hình dạng.
        </Text>
      </Stack>

      <Card withBorder padding="md" radius="md">
        <Group gap="md" align="flex-end" wrap="wrap">
          <NumberInput
            label="Số map mỗi model"
            value={count}
            onChange={(v) => setCount(Number(v) || 1)}
            min={1}
            max={100}
            w={150}
          />
          <NumberInput
            label="Width"
            value={width}
            onChange={(v) => setWidth(Number(v) || 14)}
            min={5}
            max={64}
            w={95}
          />
          <NumberInput
            label="Height"
            value={height}
            onChange={(v) => setHeight(Number(v) || 14)}
            min={5}
            max={64}
            w={95}
          />
          <NumberInput
            label="Seed"
            value={seed}
            onChange={(v) => setSeed(Number(v) || 0)}
            w={95}
          />
          <Button
            leftSection={<IconArrowsLeftRight size={16} />}
            onClick={onSubmit}
            loading={loading}
            variant="gradient"
            gradient={{ from: "cyan", to: "indigo" }}
          >
            So sánh
          </Button>
        </Group>
      </Card>

      {error && (
        <Alert icon={<IconAlertCircle size={16} />} color="red" variant="light">
          {error}
        </Alert>
      )}

      {loading && (
        <Center py="xl">
          <Loader size="md" />
        </Center>
      )}

      {data && !loading && (
        <>
          <SimpleGrid cols={{ base: 1, lg: 2 }} spacing="md">
            <Card withBorder padding="md" radius="md">
              <Group gap="xs" mb="sm">
                <IconChartBar size={18} color="var(--mantine-color-cyan-5)" />
                <Title order={5}>Bar chart so sánh</Title>
              </Group>
              <BarChart
                h={320}
                data={chartData}
                dataKey="metric"
                tickLine="y"
                gridAxis="xy"
                withLegend
                legendProps={{ verticalAlign: "bottom" }}
                series={[
                  { name: "baseline", color: "violet.6" },
                  { name: "improved", color: "teal.6" },
                ]}
              />
            </Card>

            <Card withBorder padding="md" radius="md">
              <Group gap="xs" mb="sm">
                <IconChartBar size={18} color="var(--mantine-color-cyan-5)" />
                <Title order={5}>Bảng metric trung bình</Title>
              </Group>
              <MetricsTable
                rows={[
                  { label: "Baseline", summary: data.summary.baseline, accent: "violet" },
                  { label: "Improved", summary: data.summary.improved, accent: "teal" },
                ]}
              />
            </Card>
          </SimpleGrid>

          <SimpleGrid cols={{ base: 1, lg: 2 }} spacing="md">
            <Card withBorder padding="md" radius="md">
              <Title order={5} c="violet" mb="sm">
                Baseline · {data.baseline.length} map
              </Title>
              <SimpleGrid cols={{ base: 2, sm: 3, md: 4 }} spacing="sm">
                {data.baseline.map((m) => (
                  <MapCard
                    key={m.index}
                    index={m.index}
                    grid={m.grid}
                    metrics={m.metrics}
                    maxPx={160}
                  />
                ))}
              </SimpleGrid>
            </Card>
            <Card withBorder padding="md" radius="md">
              <Title order={5} c="teal" mb="sm">
                Improved · {data.improved.length} map
              </Title>
              <SimpleGrid cols={{ base: 2, sm: 3, md: 4 }} spacing="sm">
                {data.improved.map((m) => (
                  <MapCard
                    key={m.index}
                    index={m.index}
                    grid={m.grid}
                    metrics={m.metrics}
                    maxPx={160}
                  />
                ))}
              </SimpleGrid>
            </Card>
          </SimpleGrid>
        </>
      )}

      {!data && !loading && !error && (
        <Card withBorder padding="xl" radius="md" style={{ borderStyle: "dashed" }}>
          <Center>
            <Stack align="center" gap="xs">
              <IconArrowsLeftRight size={32} color="var(--mantine-color-dimmed)" />
              <Text c="dimmed">
                Bấm "So sánh" để gen N map từ mỗi model với cùng seed
              </Text>
            </Stack>
          </Center>
        </Card>
      )}
    </Stack>
  );
}
