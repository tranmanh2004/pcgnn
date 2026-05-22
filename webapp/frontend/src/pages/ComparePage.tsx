import { useState } from "react";
import { compare, type CompareResponse } from "../api";
import { MapCard } from "../components/MapCard";
import { MetricsTable } from "../components/MetricsTable";

export function ComparePage() {
  const [count, setCount] = useState(8);
  const [seed, setSeed] = useState(0);
  const [perturb, setPerturb] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<CompareResponse | null>(null);

  async function onSubmit() {
    setLoading(true);
    setError(null);
    try {
      const result = await compare({ count, seed, perturb });
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
          <label>Số map mỗi model</label>
          <input
            type="number"
            min={1}
            max={100}
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
          {loading ? "Đang so sánh..." : "So sánh"}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {data && (
        <>
          <div className="summary-card">
            <h3>Bảng so sánh chỉ số trung bình</h3>
            <MetricsTable
              rows={[
                { label: "Baseline", summary: data.summary.baseline, accent: "var(--accent-2)" },
                { label: "Improved", summary: data.summary.improved, accent: "var(--easy)" },
              ]}
            />
          </div>

          <div className="compare-row">
            <div className="compare-column baseline">
              <h3>Baseline — {data.baseline.length} map</h3>
              <div className="grid-list">
                {data.baseline.map((m) => (
                  <MapCard
                    key={m.index}
                    index={m.index}
                    grid={m.grid}
                    metrics={m.metrics}
                    size={12}
                  />
                ))}
              </div>
            </div>
            <div className="compare-column improved">
              <h3>Improved — {data.improved.length} map</h3>
              <div className="grid-list">
                {data.improved.map((m) => (
                  <MapCard
                    key={m.index}
                    index={m.index}
                    grid={m.grid}
                    metrics={m.metrics}
                    size={12}
                  />
                ))}
              </div>
            </div>
          </div>
        </>
      )}

      {!data && !loading && !error && (
        <div className="empty">
          Bấm "So sánh" để sinh song song N map từ baseline và improved với cùng seed.
        </div>
      )}
    </div>
  );
}
