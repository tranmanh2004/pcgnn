import { useMemo, useState } from "react";
import { classify, type ClassifyResponse, type ModelName } from "../api";
import { MapCard } from "../components/MapCard";

type TierKey = "score_tier" | "range_tier" | "percentile_tier";

export function ClassifyPage() {
  const [model, setModel] = useState<ModelName>("improved");
  const [count, setCount] = useState(60);
  const [seed, setSeed] = useState(0);
  const [easyRatio, setEasyRatio] = useState(0.05);
  const [mediumRatio, setMediumRatio] = useState(0.05);
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
        perturb: true,
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

  const filtered = useMemo(() => {
    if (!data) return [];
    if (tierFilter === "all") return data.maps;
    return data.maps.filter((m) => m[tierKey] === tierFilter);
  }, [data, tierKey, tierFilter]);

  const distribution = data?.distribution[tierKey] ?? {};

  return (
    <div>
      <div className="controls">
        <div className="field">
          <label>Model</label>
          <select value={model} onChange={(e) => setModel(e.target.value as ModelName)}>
            <option value="improved">Improved</option>
            <option value="baseline">Baseline</option>
          </select>
        </div>
        <div className="field">
          <label>Số map</label>
          <input
            type="number"
            min={10}
            max={1000}
            value={count}
            onChange={(e) => setCount(Number(e.target.value))}
          />
        </div>
        <div className="field">
          <label>Seed</label>
          <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
        </div>
        <div className="field">
          <label>% Easy</label>
          <input
            type="number"
            step={0.01}
            min={0}
            max={1}
            value={easyRatio}
            onChange={(e) => setEasyRatio(Number(e.target.value))}
          />
        </div>
        <div className="field">
          <label>% Medium</label>
          <input
            type="number"
            step={0.01}
            min={0}
            max={1}
            value={mediumRatio}
            onChange={(e) => setMediumRatio(Number(e.target.value))}
          />
        </div>
        <button className="primary" onClick={onSubmit} disabled={loading}>
          {loading ? "Đang chia..." : "Chia map"}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {data && (
        <>
          <div className="summary-card">
            <h3>
              Phân bố — {data.model} · {data.count} map · seed={data.seed} · easy={data.easy_ratio}
              {" · "}medium={data.medium_ratio}
            </h3>
            <div className="field" style={{ marginBottom: 10 }}>
              <label>Cách phân tier</label>
              <select value={tierKey} onChange={(e) => setTierKey(e.target.value as TierKey)}>
                <option value="percentile_tier">Percentile (thesis, mặc định)</option>
                <option value="score_tier">Score threshold (cứng)</option>
                <option value="range_tier">Range theo metric</option>
              </select>
            </div>
            <div className="distribution">
              {Object.entries(distribution).map(([tier, n]) => (
                <span key={tier} className="chip">
                  <span className={`tier-badge ${tier}`}>{tier}</span>
                  <strong>{n}</strong>
                </span>
              ))}
            </div>
          </div>

          <div className="tier-filter">
            <button
              className={tierFilter === "all" ? "active" : ""}
              onClick={() => setTierFilter("all")}
            >
              Tất cả ({data.maps.length})
            </button>
            {Object.entries(distribution).map(([tier, n]) => (
              <button
                key={tier}
                className={tierFilter === tier ? "active" : ""}
                onClick={() => setTierFilter(tier)}
              >
                {tier} ({n})
              </button>
            ))}
          </div>

          <div className="grid-list">
            {filtered.map((m) => (
              <MapCard
                key={m.index}
                index={m.index}
                grid={m.grid}
                metrics={m.metrics}
                tier={m[tierKey]}
                size={14}
              />
            ))}
          </div>
        </>
      )}

      {!data && !loading && !error && (
        <div className="empty">
          Sinh N map, tính difficulty_score, rồi chia thành Easy/Medium/Hard theo 3 cách.
        </div>
      )}
    </div>
  );
}
