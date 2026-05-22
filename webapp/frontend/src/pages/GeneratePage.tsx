import { useState } from "react";
import {
  Button,
  Card,
  Group,
  NumberInput,
  Select,
  SimpleGrid,
  Stack,
  Text,
  Title,
  Alert,
  Loader,
  Center,
} from "@mantine/core";
import { IconAlertCircle, IconSparkles, IconChartBar } from "@tabler/icons-react";
import { generate, type GenerateResponse, type ModelName } from "../api";
import { MapCard } from "../components/MapCard";
import { MetricsTable } from "../components/MetricsTable";

export function GeneratePage() {
  const [model, setModel] = useState<ModelName>("improved");
  const [count, setCount] = useState<number>(12);
  const [seed, setSeed] = useState<number>(0);
  const [width, setWidth] = useState<number>(14);
  const [height, setHeight] = useState<number>(14);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<GenerateResponse | null>(null);

  async function onSubmit() {
    setLoading(true);
    setError(null);
    try {
      const result = await generate({ model, count, seed, width, height });
      setData(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <Stack gap="md">
      <Group justify="space-between" align="flex-end">
        <Stack gap={2}>
          <Title order={3}>Sinh map</Title>
          <Text c="dimmed" size="sm">
            Sinh N map từ checkpoint NEAT đã train. Mỗi seed cho cùng kết quả.
          </Text>
        </Stack>
      </Group>

      <Card withBorder padding="md" radius="md">
        <Group gap="md" align="flex-end" wrap="wrap">
          <Select
            label="Model"
            value={model}
            onChange={(v) => setModel((v as ModelName) ?? "improved")}
            data={[
              { value: "improved", label: "Improved (inctyseed0)" },
              { value: "baseline", label: "Baseline (neat_winner_seed0)" },
            ]}
            w={240}
          />
          <NumberInput
            label="Số lượng"
            value={count}
            onChange={(v) => setCount(Number(v) || 1)}
            min={1}
            max={200}
            w={110}
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
            leftSection={<IconSparkles size={16} />}
            onClick={onSubmit}
            loading={loading}
            variant="gradient"
            gradient={{ from: "cyan", to: "indigo" }}
          >
            Sinh map
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
          <Card withBorder padding="md" radius="md">
            <Group gap="xs" mb="sm">
              <IconChartBar size={18} color="var(--mantine-color-cyan-5)" />
              <Title order={5}>
                Tóm tắt — {data.model} · {data.count} map · {data.width}×{data.height} · seed={data.seed}
              </Title>
            </Group>
            <MetricsTable
              rows={[{ label: data.model, summary: data.summary, accent: "cyan" }]}
            />
          </Card>

          <SimpleGrid cols={{ base: 1, sm: 2, md: 3, lg: 4, xl: 5 }} spacing="md">
            {data.maps.map((m) => (
              <MapCard key={m.index} index={m.index} grid={m.grid} metrics={m.metrics} />
            ))}
          </SimpleGrid>
        </>
      )}

      {!data && !loading && !error && (
        <Card withBorder padding="xl" radius="md" style={{ borderStyle: "dashed" }}>
          <Center>
            <Stack align="center" gap="xs">
              <IconSparkles size={32} color="var(--mantine-color-dimmed)" />
              <Text c="dimmed">Chọn model + seed + số lượng, rồi bấm "Sinh map"</Text>
            </Stack>
          </Center>
        </Card>
      )}
    </Stack>
  );
}
