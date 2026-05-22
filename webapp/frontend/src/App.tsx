import { useState } from "react";
import {
  AppShell,
  Group,
  NavLink,
  Stack,
  Text,
  Title,
  Badge,
  ThemeIcon,
  Box,
} from "@mantine/core";
import {
  IconBrandReact,
  IconSparkles,
  IconArrowsLeftRight,
  IconLayoutGrid,
} from "@tabler/icons-react";
import { GeneratePage } from "./pages/GeneratePage";
import { ComparePage } from "./pages/ComparePage";
import { ClassifyPage } from "./pages/ClassifyPage";

type Tab = "generate" | "compare" | "classify";

const TABS: { id: Tab; label: string; description: string; icon: typeof IconSparkles }[] = [
  { id: "generate", label: "Sinh map", description: "Gen N map từ checkpoint", icon: IconSparkles },
  {
    id: "compare",
    label: "So sánh model",
    description: "Baseline vs improved",
    icon: IconArrowsLeftRight,
  },
  {
    id: "classify",
    label: "Chia map",
    description: "Easy / Medium / Hard",
    icon: IconLayoutGrid,
  },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("generate");

  return (
    <AppShell
      header={{ height: 64 }}
      navbar={{ width: 260, breakpoint: "sm" }}
      padding="md"
    >
      <AppShell.Header>
        <Group h="100%" px="md" justify="space-between">
          <Group gap="sm">
            <ThemeIcon size="lg" radius="md" variant="gradient" gradient={{ from: "cyan", to: "indigo" }}>
              <IconBrandReact size={20} />
            </ThemeIcon>
            <Box>
              <Title order={4} lh={1}>PCGNN Web Tool</Title>
              <Text size="xs" c="dimmed">
                14×14 maze · # wall · . floor · P player · E enemy
              </Text>
            </Box>
          </Group>
          <Group gap="xs">
            <Badge variant="light" color="cyan">FastAPI :8765</Badge>
            <Badge variant="light" color="indigo">React :5173</Badge>
          </Group>
        </Group>
      </AppShell.Header>

      <AppShell.Navbar p="sm">
        <Stack gap={4}>
          <Text size="xs" c="dimmed" tt="uppercase" fw={600} px="xs" pb="xs">
            Chức năng
          </Text>
          {TABS.map((t) => (
            <NavLink
              key={t.id}
              active={tab === t.id}
              label={t.label}
              description={t.description}
              leftSection={<t.icon size={18} />}
              onClick={() => setTab(t.id)}
              variant="filled"
            />
          ))}
        </Stack>
        <Box mt="auto">
          <Text size="xs" c="dimmed" px="xs">
            Thesis · The Knight
          </Text>
        </Box>
      </AppShell.Navbar>

      <AppShell.Main>
        {tab === "generate" && <GeneratePage />}
        {tab === "compare" && <ComparePage />}
        {tab === "classify" && <ClassifyPage />}
      </AppShell.Main>
    </AppShell>
  );
}
