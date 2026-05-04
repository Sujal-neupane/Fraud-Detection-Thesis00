import { FormEvent, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

import { predict, PredictionResponse } from "./api";
import { dashboardData } from "./analysis-data";

const DEFAULT_FEATURES = {
  TransactionAmt: 120.5,
  card1: 1111,
  C1: 0,
  D1: 12.5
};

const formatCompact = new Intl.NumberFormat("en-US", {
  notation: "compact",
  maximumFractionDigits: 1
});

const formatCurrency = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0
});

const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
const percentTick = (value: number | string) => percent(Number(value));

function App() {
  const [featuresText, setFeaturesText] = useState(
    JSON.stringify(DEFAULT_FEATURES, null, 2)
  );
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const exampleSchema = useMemo(
    () => "Expected JSON: { \"TransactionAmt\": 120.5, \"card1\": 1111 }",
    []
  );

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    setResult(null);

    let parsed: Record<string, number>;
    try {
      parsed = JSON.parse(featuresText) as Record<string, number>;
    } catch (parseError) {
      setError("Invalid JSON. Please check the input.");
      return;
    }

    setIsLoading(true);
    try {
      const prediction = await predict(parsed);
      setResult(prediction);
    } catch (requestError) {
      const message =
        requestError instanceof Error
          ? requestError.message
          : "Prediction failed";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  };

  const explanations = result?.explanations ?? [];
  const totalImpact = explanations.reduce(
    (sum, item) => sum + Math.abs(item.contribution),
    0
  );
  const chartData = explanations.map((item) => ({
    feature: item.feature,
    impact: totalImpact > 0 ? Math.abs(item.contribution) / totalImpact : 0,
    direction: item.direction
  }));

  return (
    <div className="app">
      <header className="hero">
        <div className="hero__text">
          <span className="pill">Synthetic 2019-2025 | Live analytics</span>
          <h1>Fraud Intelligence Command Center</h1>
          <p>
            Monitor fraud risk, drift signals, and operational impact in one
            view. The dashboard summarizes model performance, data health, and
            live scoring from the API.
          </p>
          <div className="hero__meta">
            <div>
              <span>Model version</span>
              <strong>{dashboardData.summary.modelVersion}</strong>
            </div>
            <div>
              <span>PR-AUC</span>
              <strong>{percent(dashboardData.summary.prAuc)}</strong>
            </div>
            <div>
              <span>Avg response</span>
              <strong>{dashboardData.summary.avgResponse}</strong>
            </div>
          </div>
        </div>
        <div className="hero__card">
          <h2>System snapshot</h2>
          <div className="snapshot">
            <div>
              <span>Data window</span>
              <strong>{dashboardData.summary.dataWindow}</strong>
            </div>
            <div>
              <span>AUC</span>
              <strong>{percent(dashboardData.summary.auc)}</strong>
            </div>
            <div>
              <span>Recall@1%</span>
              <strong>{percent(dashboardData.summary.recallAt1)}</strong>
            </div>
          </div>
          <p className="muted">
            Metrics refresh after each training run. MLflow tracks every
            experiment for reproducibility.
          </p>
        </div>
      </header>

      <section className="kpi-grid">
        {dashboardData.kpis.map((item) => (
          <div key={item.label} className="kpi-card">
            <span>{item.label}</span>
            <strong>
              {item.label === "Alert precision"
                ? percent(item.value)
                : formatCompact.format(item.value)}
            </strong>
            <em className={item.delta >= 0 ? "delta up" : "delta down"}>
              {item.delta >= 0 ? "+" : ""}
              {percent(Math.abs(item.delta))}
            </em>
          </div>
        ))}
      </section>

      <section className="grid grid--split">
        <div className="card prediction-card">
          <div className="card__header">
            <h3>Live scoring</h3>
            <span className="muted">FastAPI /predict</span>
          </div>
          <form onSubmit={handleSubmit} className="panel__form">
            <label className="panel__label" htmlFor="features">
              Features JSON
            </label>
            <textarea
              id="features"
              rows={10}
              value={featuresText}
              onChange={(event) => setFeaturesText(event.target.value)}
            />
            <span className="panel__hint">{exampleSchema}</span>

            <button type="submit" disabled={isLoading}>
              {isLoading ? "Scoring..." : "Run Prediction"}
            </button>
          </form>

          <div className="panel__output">
            {error && <p className="panel__error">{error}</p>}
            {result && (
              <div className="panel__result">
                <div
                  className={
                    result.is_fraud ? "status status--risk" : "status status--ok"
                  }
                >
                  {result.is_fraud ? "Fraud risk" : "Low risk"}
                </div>
                <p>
                  <strong>Score:</strong> {result.score.toFixed(4)}
                </p>
                <p>
                  <strong>Threshold:</strong> {result.threshold.toFixed(2)}
                </p>
                <p>
                  <strong>Model:</strong> {result.model_version}
                </p>
              </div>
            )}
          </div>
        </div>

        <div className="card chart-card">
          <div className="card__header">
            <h3>Risk explanation</h3>
            <span className="muted">Top drivers for this prediction</span>
          </div>
          {!result && (
            <p className="muted">
              Submit a transaction to see the model's top drivers and risk
              contribution breakdown.
            </p>
          )}
          {result && (
            <>
              <div className="risk-meter">
                <div className="risk-meter__label">
                  <span>Risk score</span>
                  <strong>{percent(result.score)}</strong>
                </div>
                <div className="risk-meter__bar">
                  <div
                    className="risk-meter__fill"
                    style={{ width: `${Math.min(result.score * 100, 100)}%` }}
                  />
                  <span
                    className="risk-meter__threshold"
                    style={{ left: `${result.threshold * 100}%` }}
                  />
                </div>
                <p className="muted">
                  Contributions show how each feature pushed the score above or
                  below the decision threshold.
                </p>
              </div>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={chartData} layout="vertical" margin={{ left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e3da" />
                  <XAxis tickFormatter={percentTick} type="number" />
                  <YAxis dataKey="feature" type="category" width={110} />
                  <Tooltip formatter={(value: number) => percent(value)} />
                  <Legend />
                  <Bar dataKey="impact" name="Contribution share">
                    {chartData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={
                          entry.direction === "increase" ? "#c24b33" : "#0f6931"
                        }
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </>
          )}
        </div>
      </section>
    </div>
  );
}

export default App;
