import { useState } from "react";
import { generate, type GenerateResponse, type ModelName } from "../api";
import { MapCard } from "../components/MapCard";
import { MetricsTable } from "../components/MetricsTable";

export function GeneratePage() {
  const [model, setModel] = useState<ModelName>("improved");
  const [count, setCount] = useState(12);
  const [seed, setSeed] = useState(0);
  const [perturb, setPerturb] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<GenerateResponse | null>(null);

  async function onSubmit() {
    setLoading(true);
    setError(null);
    try {
      const result = await generate({ model, count, seed, perturb });
      setData(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div className="controls">
        <div className="field">
          <label>Model</label>
          <select value={model} onChange={(e) => setModel(e.target.value as ModelName)}>
            <option value="improved">Improved (inctyseed0)</option>
            <option value="baseline">Baseline (neat_winner_seed0)</option>
          </select>
        </div>
        <div className="field">
          <label>Số lượng</label>
          <input
            type="number"
            min={1}
            max={200}
            value={count}
            onChange={(e) => setCount(Number(e.target.value))}
          />
        </div>
        <div className="field">
          <label>Seed</label>
          <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
        </div>
        <div className="field">
          <label>Perturb</label>
          <input
            type="checkbox"
            checked={perturb}
            onChange={(e) => setPerturb(e.target.checked)}
          />
        </div>
        <button className="primary" onClick={onSubmit} disabled={loading}>
          {loading ? "Đang sinh..." : "Sinh map"}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {data && (
        <>
          <div className="summary-card">
            <h3>Tóm tắt — {data.model} · {data.count} map · seed={data.seed}</h3>
            <MetricsTable
              rows={[{ label: data.model, summary: data.summary, accent: "var(--accent)" }]}
            />
          </div>
          <div className="grid-list">
            {data.maps.map((m) => (
              <MapCard key={m.index} index={m.index} grid={m.grid} metrics={m.metrics} />
            ))}
          </div>
        </>
      )}

      {!data && !loading && !error && (
        <div className="empty">Chọn model + seed + số lượng, rồi bấm "Sinh map".</div>
      )}
    </div>
  );
}
