import { useState } from "react";
import { GeneratePage } from "./pages/GeneratePage";
import { ComparePage } from "./pages/ComparePage";
import { ClassifyPage } from "./pages/ClassifyPage";

type Tab = "generate" | "compare" | "classify";

const TABS: { id: Tab; label: string }[] = [
  { id: "generate", label: "Sinh map" },
  { id: "compare", label: "So sánh baseline vs improved" },
  { id: "classify", label: "Chia map theo độ khó" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("generate");

  return (
    <div className="app">
      <div className="header">
        <h1>PCGNN Web Tool</h1>
        <span className="subtitle">
          14×14 maze · # wall · . floor · P player · E enemy
        </span>
      </div>
      <div className="tabs">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={`tab ${tab === t.id ? "active" : ""}`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>
      {tab === "generate" && <GeneratePage />}
      {tab === "compare" && <ComparePage />}
      {tab === "classify" && <ClassifyPage />}
    </div>
  );
}
