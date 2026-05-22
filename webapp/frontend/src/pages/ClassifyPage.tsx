import { useMemo, useState } from "react";
import {
  Alert,
  Badge,
  Button,
  Card,
  Center,
  Chip,
  Group,
  Loader,
  NumberInput,
  Select,
  SimpleGrid,
  Stack,
  Text,
  Title,
} from "@mantine/core";
import { DonutChart } from "@mantine/charts";
import {
  IconAlertCircle,
  IconChartPie,
  IconLayoutGrid,
} from "@tabler/icons-react";
import { classify, type ClassifyResponse, type ModelName } from "../api";
import { MapCard } from "../components/MapCard";

type TierKey = "score_tier" | "range_tier" | "percentile_tier";

const TIER_COLORS: Record<string, string> = {
  easy: "teal.6",
  medium: "yellow.6",
  hard: "red.6",
  unclassified: "gray.6",
};

const TIER_BADGE_COLORS: Record<string, string> = {
  easy: "teal",
  medium: "yellow",
  hard: "red",
  unclassified: "gray",
};

export function ClassifyPage() {
  const [model, setModel] = useState<ModelName>("improved");
  const [count, setCount] = useState<number>(60);
  const [seed, setSeed] = useState<number>(0);
  const [width, setWidth] = useState<number>(14);
  const [height, setHeight] = useState<number>(14);
  const [easyRatio, setEasyRatio] = useState<number>(0.05);
  const [mediumRatio, setMediumRatio] = useState<number>(0.05);
  const [tierKey, setTierKey] = useState<TierKey>("percentile_tier");
  const [tierFilter, setTierFilter] = useState<string>("all");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<ClassifyResponse | null>(null);

  async function onSubmit() {
    setLoading(true);
    setError(null);
    try {
      const result = await classify({
        model,
        count,
        seed,
        width,
        height,
        easy_ratio: easyRatio,
        medium_ratio: mediumRatio,
      });
      setData(result);
      setTierFilter("all");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  const distribution = data?.distribution[tierKey] ?? {};

  const donutData = useMemo(
    () =>
      Object.entries(distribution).map(([name, value]) => ({
        name,
        value,
        color: TIER_COLORS[name] ?? "gray.6",
      })),
    [distribution]
  );

  const filtered = useMemo(() => {
    if (!data) return [];
    if (tierFilter === "all") return data.maps;
    return data.maps.filter((m) => m[tierKey] === tierFilter);
  }, [data, tierKey, tierFilter]);

  return (
    <Stack gap="md">
      <Stack gap={2}>
        <Title order={3}>Chia map theo độ khó</Title>
        <Text c="dimmed" size="sm">
          Sinh N map → tính difficulty_score → phân Easy / Medium / Hard theo 3 cách
          (percentile là cách thesis dùng chính thức).
        </Text>
      </Stack>

      <Card withBorder padding="md" radius="md">
        <Group gap="md" align="flex-end" wrap="wrap">
          <Select
            label="Model"
            value={model}
            onChange={(v) => setModel((v as ModelName) ?? "improved")}
            data={[
              { value: "improved", label: "Improved" },
              { value: "baseline", label: "Baseline" },
            ]}
            w={180}
          />
          <NumberInput
            label="Số map"
            value={count}
            onChange={(v) => setCount(Number(v) || 10)}
            min={10}
            max={1000}
            w={120}
          />
          <NumberInput
            label="Seed"
            value={seed}
            onChange={(v) => setSeed(Number(v) || 0)}
            w={100}
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
            label="% Easy"
            value={easyRatio}
            onChange={(v) => setEasyRatio(Number(v) || 0)}
            step={0.01}
            min={0}
            max={1}
            decimalScale={3}
            w={110}
          />
          <NumberInput
            label="% Medium"
            value={mediumRatio}
            onChange={(v) => setMediumRatio(Number(v) || 0)}
            step={0.01}
            min={0}
            max={1}
            decimalScale={3}
            w={110}
          />
          <Button
            leftSection={<IconLayoutGrid size={16} />}
            onClick={onSubmit}
            loading={loading}
            variant="gradient"
            gradient={{ from: "cyan", to: "indigo" }}
          >
            Chia map
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
          <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
            <Card withBorder padding="md" radius="md">
              <Group gap="xs" mb="sm" justify="space-between">
                <Group gap="xs">
                  <IconChartPie size={18} color="var(--mantine-color-cyan-5)" />
                  <Title order={5}>Phân bố tier</Title>
                </Group>
                <Select
                  size="xs"
                  value={tierKey}
                  onChange={(v) => setTierKey((v as TierKey) ?? "percentile_tier")}
                  data={[
                    { value: "percentile_tier", label: "Percentile (thesis)" },
                    { value: "score_tier", label: "Score threshold" },
                    { value: "range_tier", label: "Range theo metric" },
                  ]}
                  w={200}
                />
              </Group>
              <Center>
                <DonutChart
                  h={240}
                  data={donutData}
                  withLabelsLine
                  withLabels
                  chartLabel={`${data.count} map`}
                />
              </Center>
            </Card>

            <Card withBorder padding="md" radius="md">
              <Title order={5} mb="sm">
                Thông tin
              </Title>
              <Stack gap={6}>
                <Group justify="space-between">
                  <Text c="dimmed">Model</Text>
                  <Badge variant="light">{data.model}</Badge>
                </Group>
                <Group justify="space-between">
                  <Text c="dimmed">Số map</Text>
                  <Text fw={600}>{data.count}</Text>
                </Group>
                <Group justify="space-between">
                  <Text c="dimmed">Seed</Text>
                  <Text fw={600}>{data.seed}</Text>
                </Group>
                <Group justify="space-between">
                  <Text c="dimmed">% Easy / Medium</Text>
                  <Text fw={600}>
                    {data.easy_ratio} / {data.medium_ratio}
                  </Text>
                </Group>
                <Group justify="space-between" mt="xs" pt="xs" style={{ borderTop: "1px solid var(--mantine-color-dark-4)" }}>
                  <Text c="dimmed">Cách phân tier</Text>
                  <Badge variant="light" color="cyan">{tierKey}</Badge>
                </Group>
                <Group gap={6} mt={4}>
                  {Object.entries(distribution).map(([tier, n]) => (
                    <Badge
                      key={tier}
                      color={TIER_BADGE_COLORS[tier] ?? "gray"}
                      variant="filled"
                    >
                      {tier}: {n}
                    </Badge>
                  ))}
                </Group>
              </Stack>
            </Card>
          </SimpleGrid>

          <Card withBorder padding="md" radius="md">
            <Group justify="space-between" mb="sm">
              <Title order={5}>Map đã phân loại</Title>
              <Chip.Group
                value={tierFilter}
                onChange={(v) => setTierFilter(v as string)}
              >
                <Group gap={6}>
                  <Chip value="all" size="xs">
                    Tất cả ({data.maps.length})
                  </Chip>
                  {Object.entries(distribution).map(([tier, n]) => (
                    <Chip key={tier} value={tier} size="xs" color={TIER_BADGE_COLORS[tier]}>
                      {tier} ({n})
                    </Chip>
                  ))}
                </Group>
              </Chip.Group>
            </Group>
            <SimpleGrid cols={{ base: 2, sm: 3, md: 4, lg: 5, xl: 6 }} spacing="sm">
              {filtered.map((m) => (
                <MapCard
                  key={m.index}
                  index={m.index}
                  grid={m.grid}
                  metrics={m.metrics}
                  tier={m[tierKey]}
                  maxPx={160}
                />
              ))}
            </SimpleGrid>
          </Card>
        </>
      )}

      {!data && !loading && !error && (
        <Card withBorder padding="xl" radius="md" style={{ borderStyle: "dashed" }}>
          <Center>
            <Stack align="center" gap="xs">
              <IconLayoutGrid size={32} color="var(--mantine-color-dimmed)" />
              <Text c="dimmed">
                Sinh N map, tính difficulty_score, rồi chia Easy / Medium / Hard
              </Text>
            </Stack>
          </Center>
        </Card>
      )}
    </Stack>
  );
}
