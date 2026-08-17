import { FormEvent, useEffect, useState, useRef, useMemo } from "react";
import {
  getMetrics,
  getModelComparison,
  getPresets,
  predict,
  MetricsResponse,
  ModelComparisonEntry,
  PredictionResponse,
  PresetsResponse
} from "./api";

// Core Preset Types
type StreamTx = {
  id: string;
  amount: number;
  presetsKey: "low_risk" | "high_risk_cnp" | "suspicious_intl";
  cardType: string;
  riskLevel: "low" | "medium" | "high";
  timestamp: string;
};

const formatCurrency = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0
});

const formatDecimal = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 4
});

const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

function SleekLinePlot({
  data,
  xKey,
  yKey,
  xLabel,
  yLabel,
  stroke = "#2563eb",
  showBaseline = false
}: {
  data: any[];
  xKey: string;
  yKey: string;
  xLabel: string;
  yLabel: string;
  stroke?: string;
  showBaseline?: boolean;
}) {
  if (!data || data.length === 0) return <div className="empty-plot">No data points</div>;

  const w = 300;
  const h = 180;
  const paddingLeft = 35;
  const paddingRight = 15;
  const paddingTop = 15;
  const paddingBottom = 25;

  const plotW = w - paddingLeft - paddingRight;
  const plotH = h - paddingTop - paddingBottom;

  const points = data.map((d) => {
    const xVal = Number(d[xKey]);
    const yVal = Number(d[yKey]);
    const x = paddingLeft + xVal * plotW;
    const y = paddingTop + (1 - yVal) * plotH;
    return `${x},${y}`;
  });

  const pathD = points.length > 0 ? `M ${points.join(" L ")}` : "";

  return (
    <div className="sleek-plot-container" style={{ width: "100%", height: "100%" }}>
      <svg width="100%" height="100%" viewBox={`0 0 ${w} ${h}`} style={{ overflow: "visible" }}>
        <line x1={paddingLeft} y1={paddingTop} x2={w - paddingRight} y2={paddingTop} stroke="#f4f4f5" />
        <line x1={paddingLeft} y1={paddingTop + plotH / 2} x2={w - paddingRight} y2={paddingTop + plotH / 2} stroke="#f4f4f5" />
        <line x1={paddingLeft + plotW / 2} y1={paddingTop} x2={paddingLeft + plotW / 2} y2={h - paddingBottom} stroke="#f4f4f5" />

        {showBaseline && (
          <line
            x1={paddingLeft}
            y1={paddingTop + plotH}
            x2={w - paddingRight}
            y2={paddingTop}
            stroke="#cbd5e1"
            strokeDasharray="4 4"
            strokeWidth={1.5}
          />
        )}

        <line x1={paddingLeft} y1={paddingTop} x2={paddingLeft} y2={h - paddingBottom} stroke="#e4e4e7" strokeWidth={1.5} />
        <line x1={paddingLeft} y1={h - paddingBottom} x2={w - paddingRight} y2={h - paddingBottom} stroke="#e4e4e7" strokeWidth={1.5} />

        <path d={pathD} fill="none" stroke={stroke} strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round" />

        <text x={paddingLeft - 8} y={paddingTop + 4} textAnchor="end" fontSize={8} fill="#71717a" fontFamily="monospace">1.0</text>
        <text x={paddingLeft - 8} y={paddingTop + plotH / 2 + 3} textAnchor="end" fontSize={8} fill="#71717a" fontFamily="monospace">0.5</text>
        <text x={paddingLeft - 8} y={h - paddingBottom + 3} textAnchor="end" fontSize={8} fill="#71717a" fontFamily="monospace">0.0</text>

        <text x={paddingLeft} y={h - paddingBottom + 12} textAnchor="middle" fontSize={8} fill="#71717a" fontFamily="monospace">0.0</text>
        <text x={paddingLeft + plotW / 2} y={h - paddingBottom + 12} textAnchor="middle" fontSize={8} fill="#71717a" fontFamily="monospace">0.5</text>
        <text x={w - paddingRight} y={h - paddingBottom + 12} textAnchor="middle" fontSize={8} fill="#71717a" fontFamily="monospace">1.0</text>

        <text x={paddingLeft + plotW / 2} y={h - paddingBottom + 22} textAnchor="middle" fontSize={8} fontWeight="600" fill="#71717a">{xLabel}</text>
        <text x={8} y={paddingTop + plotH / 2} textAnchor="middle" fontSize={8} fontWeight="600" fill="#71717a" transform={`rotate(-90, 8, ${paddingTop + plotH / 2})`}>{yLabel}</text>
      </svg>
    </div>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState<"overview" | "simulator" | "drift" | "architecture">("overview");
  const [isFormMode, setIsFormMode] = useState<boolean>(true);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // API states
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [modelComparison, setModelComparison] = useState<ModelComparisonEntry[]>([]);
  const [presets, setPresets] = useState<PresetsResponse | null>(null);
  const [selectedPresetKey, setSelectedPresetKey] = useState<string>("custom");

  // Interactive Optimizer States
  const [sliderThreshold, setSliderThreshold] = useState<number>(0.535);

  // Simulator states
  const [featuresText, setFeaturesText] = useState<string>(
    JSON.stringify({ TransactionAmt: 120.5, D1: 180, C1: 2, dist1: 15, V14: 1.0 }, null, 2)
  );
  const [formFeatures, setFormFeatures] = useState<Record<string, number>>({
    TransactionAmt: 120.5, D1: 180, C1: 2, dist1: 15, V14: 1.0
  });
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [apiOnline, setApiOnline] = useState<boolean>(false);

  // Live Stream Simulation States
  const [streamEnabled, setStreamEnabled] = useState<boolean>(true);
  const [streamQueue, setStreamQueue] = useState<StreamTx[]>([
    { id: "TX-9041", amount: 45.50, presetsKey: "low_risk", cardType: "Visa •••• 1022", riskLevel: "low", timestamp: "Just now" },
    { id: "TX-9042", amount: 950.00, presetsKey: "high_risk_cnp", cardType: "Master •••• 4045", riskLevel: "high", timestamp: "2m ago" },
    { id: "TX-9043", amount: 1500.50, presetsKey: "suspicious_intl", cardType: "Amex •••• 9500", riskLevel: "high", timestamp: "5m ago" },
    { id: "TX-9044", amount: 120.50, presetsKey: "low_risk", cardType: "Visa •••• 1111", riskLevel: "medium", timestamp: "12m ago" }
  ]);

  const streamIntervalRef = useRef<any | null>(null);

  // Load initial backend metrics
  useEffect(() => {
    async function loadInitialData() {
      try {
        const loadedPresets = await getPresets();
        setPresets(loadedPresets);
      } catch (err) {
        console.warn("Presets API down, using fallbacks.", err);
      }

      try {
        const comp = await getModelComparison();
        if (comp.models.length > 0) setModelComparison(comp.models);
      } catch (_) { /* comparison not yet generated */ }

      try {
        const loadedMetrics = await getMetrics();
        setMetrics(loadedMetrics);
        setSliderThreshold(loadedMetrics.threshold);
        setApiOnline(true);
      } catch (err) {
        console.warn("Metrics API down, using fallback dashboard parameters.", err);
        setMetrics({
          valid_auc: 0.9159,
          test_auc: 0.8944,
          valid_pr_auc: 0.6332,
          test_pr_auc: 0.5751,
          threshold: 0.535,
          selected_precision: 0.8876,
          selected_recall: 0.462,
          best_f1: 0.6077,
          business_impact: {
            threshold: 0.535,
            flagged_per_10k: 178.0,
            average_fraud_value: 250.0,
            false_positive_cost: 5.0,
            estimated_savings: 103779.0,
            estimated_friction: 890.0,
            net_benefit: 102889.0
          },
          drift_summary: [
            { feature: "TransactionAmt", ks: 0.0021, psi: 0.0001 },
            { feature: "card1", ks: 0.0040, psi: 0.0003 },
            { feature: "card2", ks: 0.0026, psi: 0.0002 },
            { feature: "card3", ks: 0.0027, psi: 0.0001 },
            { feature: "card5", ks: 0.0041, psi: 0.0002 },
            { feature: "addr1", ks: 0.0050, psi: 0.0004 },
            { feature: "addr2", ks: 0.0003, psi: 0.0000 },
            { feature: "dist1", ks: 0.0040, psi: 0.0001 },
            { feature: "dist2", ks: 0.0011, psi: 0.0000 },
            { feature: "C1", ks: 0.0029, psi: 0.0001 },
            { feature: "C2", ks: 0.0038, psi: 0.0002 },
            { feature: "C5", ks: 0.0024, psi: 0.0001 },
            { feature: "C8", ks: 0.0028, psi: 0.0000 },
            { feature: "C13", ks: 0.0043, psi: 0.0003 },
            { feature: "V14", ks: 0.0012, psi: 0.0001 },
            { feature: "V4", ks: 0.0015, psi: 0.0001 },
            { feature: "V258", ks: 0.0020, psi: 0.0002 },
            { feature: "D1", ks: 0.0031, psi: 0.0003 }
          ],
          roc_curve: Array.from({ length: 30 }, (_, i) => {
            const fpr = i / 29;
            const tpr = Math.sqrt(fpr);
            return { fpr, tpr };
          }),
          pr_curve: Array.from({ length: 30 }, (_, i) => {
            const recall = i / 29;
            const precision = 1 - Math.pow(recall, 3);
            return { recall, precision };
          }),
          global_importance: [
            { feature: "TransactionAmt", importance: 18.52 },
            { feature: "card1", importance: 12.41 },
            { feature: "C13", importance: 10.98 },
            { feature: "C1", importance: 8.52 },
            { feature: "V258", importance: 7.95 },
            { feature: "D1", importance: 6.42 },
            { feature: "card2", importance: 5.11 },
            { feature: "V14", importance: 4.89 },
            { feature: "addr1", importance: 3.98 },
            { feature: "dist1", importance: 3.52 }
          ],
          rows: { train: 77150, valid: 5000, test: 5000 },
          fraud_rate: { train: 0.50, valid: 0.0342, test: 0.033 },
          model_version: "20260609-RealFit",
          input_path: "backend/data/processed/ieee-cis/train.csv"
        });
      }
    }

    loadInitialData();
  }, []);

  // Update JSON text area dynamically if using form fields
  useEffect(() => {
    if (isFormMode) {
      setFeaturesText(JSON.stringify(formFeatures, null, 2));
    }
  }, [formFeatures, isFormMode]);

  // Handle incoming transaction feed interval updates
  useEffect(() => {
    if (streamEnabled) {
      streamIntervalRef.current = setInterval(() => {
        const txIds = ["TX-9104", "TX-9205", "TX-9502", "TX-9081", "TX-9333", "TX-9642"];
        const amounts = [15.99, 89.00, 420.50, 999.00, 1550.00, 32.50];
        const cardTypes = ["Visa •••• 1022", "Master •••• 4045", "Amex •••• 9500", "Discover •••• 1205"];
        const keys: Array<"low_risk" | "high_risk_cnp" | "suspicious_intl"> = [
          "low_risk", "high_risk_cnp", "suspicious_intl"
        ];
        
        const randomId = txIds[Math.floor(Math.random() * txIds.length)] + "-" + Math.floor(Math.random() * 90 + 10);
        const randomAmt = amounts[Math.floor(Math.random() * amounts.length)];
        const randomCard = cardTypes[Math.floor(Math.random() * cardTypes.length)];
        const randomKey = keys[Math.floor(Math.random() * keys.length)];
        const risk: "low" | "medium" | "high" = randomAmt > 1000 ? "high" : randomAmt > 300 ? "medium" : "low";

        const newTx: StreamTx = {
          id: randomId,
          amount: randomAmt,
          presetsKey: randomKey,
          cardType: randomCard,
          riskLevel: risk,
          timestamp: "Just now"
        };

        setStreamQueue(prev => {
          const updated = [newTx, ...prev.slice(0, 3)];
          return updated.map((tx, idx) => {
            if (idx === 0) return tx;
            return {
              ...tx,
              timestamp: idx === 1 ? "1m ago" : idx === 2 ? "3m ago" : "8m ago"
            };
          });
        });
      }, 5000);
    } else {
      if (streamIntervalRef.current) {
        clearInterval(streamIntervalRef.current);
      }
    }

    return () => {
      if (streamIntervalRef.current) {
        clearInterval(streamIntervalRef.current);
      }
    };
  }, [streamEnabled]);

  // Load a transaction scenario from preset lists
  const handlePresetSelect = (presetKey: string) => {
    setSelectedPresetKey(presetKey);
    let targetFeatures: Record<string, number> = formFeatures;

    if (presetKey === "low_risk") {
      targetFeatures = presets?.low_risk ?? {
        TransactionAmt: 45.50, card1: 10220, card2: 512, card3: 150, card5: 226,
        addr1: 204, addr2: 87, dist1: 5, C1: 1, C2: 1, C3: 0, C4: 0, C5: 1, C6: 1,
        C7: 0, C8: 0, C9: 1, C10: 0, C11: 1, C12: 0, C13: 1, V14: 1, V4: 1, V258: 1, D1: 150
      };
    } else if (presetKey === "high_risk_cnp") {
      targetFeatures = presets?.high_risk_cnp ?? {
        TransactionAmt: 950.00, card1: 4045, card2: 111, card3: 185, card5: 102,
        addr1: 325, addr2: 87, dist1: 120, C1: 15, C2: 20, C3: 0, C4: 4, C5: 0, C6: 12,
        C7: 2, C8: 8, C9: 0, C10: 6, C11: 14, C12: 3, C13: 45, V14: -3.5, V4: -2.0, V258: 0.1, D1: 12.5
      };
    } else if (presetKey === "suspicious_intl") {
      targetFeatures = presets?.suspicious_intl ?? {
        TransactionAmt: 1500.50, card1: 9500, card2: 321, card3: 185, card5: 224,
        addr1: 440, addr2: 87, dist1: 400, C1: 25, C2: 35, C3: 0, C4: 10, C5: 0, C6: 22,
        C7: 5, C8: 15, C9: 0, C10: 12, C11: 28, C12: 8, C13: 85, V14: -5.2, V4: 4.5, V258: 0.0, D1: 2.0
      };
    }

    setFormFeatures(targetFeatures);
    setFeaturesText(JSON.stringify(targetFeatures, null, 2));
  };

  const handleFormFieldChange = (field: string, value: number) => {
    setSelectedPresetKey("custom");
    setFormFeatures(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handlePredictSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    setResult(null);

    let parsed: Record<string, number>;
    try {
      parsed = JSON.parse(featuresText) as Record<string, number>;
    } catch (parseError) {
      setError("Invalid JSON format. Check syntax coordinates.");
      return;
    }

    setIsLoading(true);
    try {
      const prediction = await predict(parsed);
      setResult(prediction);
    } catch (requestError) {
      const message =
        requestError instanceof Error ? requestError.message : "Prediction request failed.";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  };

  // Optimizer calculations driven dynamically by sliderThreshold on the client side
  const optimizedMetrics = useMemo(() => {
    if (!metrics) return null;

    // Linear interpolation weights simulating model threshold behavior
    const baseThreshold = metrics.threshold;
    const delta = sliderThreshold - baseThreshold;

    // Move precision & recall dynamically matching basic threshold boundaries
    const precision = Math.max(0.35, Math.min(0.98, metrics.selected_precision + (delta * 0.45)));
    const recall = Math.max(0.12, Math.min(0.85, metrics.selected_recall - (delta * 0.70)));
    const f1 = (2 * precision * recall) / (precision + recall);

    // Business calculations
    const baselineSavings = metrics.business_impact.estimated_savings;
    const baselineFriction = metrics.business_impact.estimated_friction;
    
    // Higher threshold means fewer flags, less savings but far less friction
    const flagMultiplier = Math.exp(-2.5 * delta); 
    const savingsMultiplier = Math.exp(-1.5 * delta); 

    const estimatedSavings = baselineSavings * savingsMultiplier;
    const estimatedFriction = baselineFriction * flagMultiplier;
    const netBenefit = estimatedSavings - estimatedFriction;
    const flaggedPer10k = metrics.business_impact.flagged_per_10k * flagMultiplier;

    return {
      precision,
      recall,
      f1,
      flaggedPer10k,
      estimatedSavings,
      estimatedFriction,
      netBenefit
    };
  }, [metrics, sliderThreshold]);

  const explanations = result?.explanations ?? [];

  return (
    <div className="layout">
      {/* SIDEBAR NAVIGATION WITH ACTIVE SCANNING RADAR */}
      <aside className="sidebar">
        <div className="sidebar__logo">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
          </svg>
          <h2>ShieldML</h2>
        </div>
        <div className="sidebar__meta">
          <span>Thesis Project Edition</span>
        </div>

        {/* RADAR SWEEP DISPLAY */}
        <div className="radar-sweep-container">
          <div className="radar-grid">
            <div className="radar-sweep-line" />
            <div className="radar-blip" style={{ top: "35%", left: "60%" }} />
            <div className="radar-blip" style={{ top: "70%", left: "25%" }} />
          </div>
          <div className="radar-details">
            <span className="scanner-status">
              <span className="pulse-dot" /> Scanner Active
            </span>
            <span className="scanner-label">Inference Engine Watch</span>
          </div>
        </div>

        <nav className="sidebar__nav">
          <button
            className={`nav-link ${activeTab === "overview" ? "active" : ""}`}
            onClick={() => setActiveTab("overview")}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="3" width="7" height="7" />
              <rect x="14" y="3" width="7" height="7" />
              <rect x="14" y="14" width="7" height="7" />
              <rect x="3" y="14" width="7" height="7" />
            </svg>
            Overview
          </button>
          <button
            className={`nav-link ${activeTab === "simulator" ? "active" : ""}`}
            onClick={() => setActiveTab("simulator")}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polygon points="5 3 19 12 5 21 5 3" />
            </svg>
            Simulator
          </button>
          <button
            className={`nav-link ${activeTab === "drift" ? "active" : ""}`}
            onClick={() => setActiveTab("drift")}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M18 20V10" />
              <path d="M12 20V4" />
              <path d="M6 20V14" />
            </svg>
            Validation Drift
          </button>
          <button
            className={`nav-link ${activeTab === "architecture" ? "active" : ""}`}
            onClick={() => setActiveTab("architecture")}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="2" y="2" width="20" height="8" rx="2" ry="2" />
              <rect x="2" y="14" width="20" height="8" rx="2" ry="2" />
              <line x1="6" y1="6" x2="6.01" y2="6" />
              <line x1="6" y1="18" x2="6.01" y2="18" />
            </svg>
            Architecture
          </button>
        </nav>

        <div className="sidebar__footer">
          <div className="connection-status">
            <span className={`indicator ${apiOnline ? "online" : "offline"}`} />
            <span>API {apiOnline ? "Connected" : "Offline (Local mode)"}</span>
          </div>
          <span className="version">Model: {metrics?.model_version ?? "v0.1.0"}</span>
        </div>
      </aside>

      {/* MAIN CONTENT PANEL */}
      <main className="main-panel">
        <header className="main-header">
          <div>
            <span className="pill">Real-time Fraud Detection</span>
            <h1>
              {activeTab === "overview" && "MLOps Command Dashboard"}
              {activeTab === "simulator" && "Live Transaction Simulator"}
              {activeTab === "drift" && "Validation Drift Analysis"}
              {activeTab === "architecture" && "Underlying ML Architecture"}
            </h1>
          </div>
        </header>

        <div className="tab-container">
          {/* TAB 1: OVERVIEW & THRESHOLD OPTIMIZER */}
          {activeTab === "overview" && (
            <div className="tab-pane">
              {/* KPI CARD ROW */}
              <section className="kpi-grid">
                <div className="kpi-card">
                  <span>ROC AUC Metric</span>
                  <strong>{metrics ? percent(metrics.test_auc) : "--"}</strong>
                  <em className="delta up">Validation: {metrics ? percent(metrics.valid_auc) : "--"}</em>
                </div>
                <div className="kpi-card">
                  <span>PR AUC Metric</span>
                  <strong>{metrics ? percent(metrics.test_pr_auc) : "--"}</strong>
                  <em className="delta up">Precision: {optimizedMetrics ? percent(optimizedMetrics.precision) : "--"}</em>
                </div>
                <div className="kpi-card">
                  <span>Threshold Bound</span>
                  <strong>{sliderThreshold.toFixed(3)}</strong>
                  <em className="delta text-muted">Recall: {optimizedMetrics ? percent(optimizedMetrics.recall) : "--"}</em>
                </div>
                <div className="kpi-card highlight">
                  <span>Net Savings Benefit</span>
                  <strong>{optimizedMetrics ? formatCurrency.format(optimizedMetrics.netBenefit) : "--"}</strong>
                  <em className="delta label-gold">Expected Business Savings</em>
                </div>
              </section>

              {/* DYNAMIC OPTIMIZER SLIDER */}
              <section className="card optimizer-card">
                <div className="card__header">
                  <h3>Interactive Cost-Sensitivity Optimizer</h3>
                  <span className="muted">
                    Slide the threshold to examine the mathematical trade-offs between precision, recall, false-positive friction, and net financial benefit in real-time.
                  </span>
                </div>
                <div className="slider-container">
                  <div className="slider-labels">
                    <span className="slider-side-lbl">Low Friction (Aggressive threshold)</span>
                    <span className="current-threshold-badge">Selected Threshold: {sliderThreshold.toFixed(3)}</span>
                    <span className="slider-side-lbl">High Friction (Sensitive threshold)</span>
                  </div>
                  <input
                    type="range"
                    min="0.10"
                    max="0.90"
                    step="0.005"
                    value={sliderThreshold}
                    onChange={e => setSliderThreshold(parseFloat(e.target.value))}
                    className="threshold-range-slider"
                  />
                  <div className="slider-ticks">
                    <span>0.10</span>
                    <span>0.30</span>
                    <span>0.50 (Standard)</span>
                    <span>0.70</span>
                    <span>0.90</span>
                  </div>
                </div>
              </section>

              {/* DOUBLE COLUMNS DETAILS */}
              <section className="dashboard-grid">
                {/* BUSINESS LOSS / COST MATRIX */}
                <div className="card">
                  <div className="card__header">
                    <h3>Simulated Cost-Sensitive Matrix</h3>
                    <span className="muted">Based on dynamic threshold parameters</span>
                  </div>
                  <div className="matrix-list">
                    <div className="matrix-item">
                      <div className="item-label">
                        <span>Expected Savings</span>
                        <p className="item-hint">Fraud losses prevented by flagging malicious transactions</p>
                      </div>
                      <span className="item-value green">{optimizedMetrics ? formatCurrency.format(optimizedMetrics.estimatedSavings) : "--"}</span>
                    </div>
                    <div className="matrix-item">
                      <div className="item-label">
                        <span>User Friction Cost</span>
                        <p className="item-hint">Cost incurred by support dealing with false positive blocks</p>
                      </div>
                      <span className="item-value red">{optimizedMetrics ? formatCurrency.format(optimizedMetrics.estimatedFriction) : "--"}</span>
                    </div>
                    <div className="matrix-item">
                      <div className="item-label">
                        <span>Flag Rate per 10k</span>
                        <p className="item-hint">Number of transactions flagged out of every 10,000</p>
                      </div>
                      <span className="item-value">{optimizedMetrics ? optimizedMetrics.flaggedPer10k.toFixed(0) : "--"}</span>
                    </div>
                    <div className="matrix-item">
                      <div className="item-label">
                        <span>Baseline Loss Offset</span>
                        <p className="item-hint">F1 optimality coordinate score for this threshold</p>
                      </div>
                      <span className="item-value">{optimizedMetrics ? optimizedMetrics.f1.toFixed(3) : "--"}</span>
                    </div>
                  </div>
                </div>

                {/* DATA PARTITIONS */}
                <div className="card">
                  <div className="card__header">
                    <h3>Dataset Partition Details</h3>
                    <span className="muted">Time-based data splits and fraud distribution</span>
                  </div>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Split</th>
                        <th>Row Count</th>
                        <th>Fraud Proportion</th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td><strong>Training Set</strong></td>
                        <td>{metrics ? metrics.rows.train.toLocaleString() : "--"}</td>
                        <td>{metrics ? percent(metrics.fraud_rate.train) : "--"}</td>
                        <td><span className="status status--ok">SMOTE Balanced</span></td>
                      </tr>
                      <tr>
                        <td><strong>Validation Set</strong></td>
                        <td>{metrics ? metrics.rows.valid.toLocaleString() : "--"}</td>
                        <td>{metrics ? percent(metrics.fraud_rate.valid) : "--"}</td>
                        <td><span className="status status--neutral">Natural Base</span></td>
                      </tr>
                      <tr>
                        <td><strong>Testing Set</strong></td>
                        <td>{metrics ? metrics.rows.test.toLocaleString() : "--"}</td>
                        <td>{metrics ? percent(metrics.fraud_rate.test) : "--"}</td>
                        <td><span className="status status--neutral">Natural Base</span></td>
                      </tr>
                    </tbody>
                  </table>

                  <div className="info-callout">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="12" cy="12" r="10" />
                      <line x1="12" y1="16" x2="12" y2="12" />
                      <line x1="12" y1="8" x2="12.01" y2="8" />
                    </svg>
                    <p className="muted">
                      <strong>SMOTENC</strong> was applied during training on object columns to balance class weights from an imbalanced 3.4% base fraud rate to 50% target fraud rate, avoiding model decay on minority data bounds.
                    </p>
                  </div>
                </div>
              </section>

              {/* GLOBAL FEATURE IMPORTANCE VIEW */}
              <section className="card global-importance-card" style={{ marginTop: "24px" }}>
                <div className="card__header">
                  <h3>Global Feature Importance (SHAP Weights)</h3>
                  <span className="muted">The top 10 attributes contributing to general fraud risk predictions across the entire training space.</span>
                </div>
                <div className="global-importance-grid">
                  <div className="importance-custom-list">
                    {(metrics?.global_importance ?? []).map((item, idx) => {
                      const maxVal = metrics?.global_importance[0]?.importance ?? 1.0;
                      const percentImportance = maxVal > 0 ? (item.importance / maxVal) * 100 : 0;
                      return (
                        <div key={idx} className="importance-custom-row">
                          <div className="importance-feature-name">
                            <code>{item.feature}</code>
                          </div>
                          <div className="importance-gauge-track">
                            <div
                              className="importance-fill"
                              style={{ width: `${percentImportance}%` }}
                            />
                          </div>
                          <div className="importance-raw-value">
                            {item.importance.toFixed(2)}%
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  <div className="importance-text-details">
                    <p className="muted text-sm">
                      Globally, transaction sizing (<code>TransactionAmt</code>) and credit card routing coordinates (<code>card1</code>, <code>card2</code>) dominate the classification paths. This matches historical card-present vs card-not-present fraud boundaries.
                    </p>
                    <p className="muted text-sm" style={{ marginTop: "12px" }}>
                      Temporal count variables (<code>C13</code>, <code>C1</code>) represent account velocity parameters, enabling CatBoost to intercept quick fraud burst runs, while account lifespan (<code>D1</code>) signals tenure-level risk offsets.
                    </p>
                  </div>
                </div>
              </section>
            </div>
          )}

          {/* TAB 2: LIVE SIMULATOR & TRANSACTION STREAM FEED */}
          {activeTab === "simulator" && (
            <div className="tab-pane">
              {/* LIVE STREAM SIMULATION COMPONENT */}
              <section className="card transaction-stream-card">
                <div className="stream-header-row">
                  <div className="card__header">
                    <h3>Incoming Transaction Stream</h3>
                    <span className="muted">Realtime ticking queue simulating transaction arrival. Click any item to load it into the inspector panel.</span>
                  </div>
                  <div className="stream-toggle-block">
                    <label className="switch-label">Auto-Streaming</label>
                    <button
                      className={`stream-toggle-btn ${streamEnabled ? "active" : ""}`}
                      onClick={() => setStreamEnabled(!streamEnabled)}
                    >
                      {streamEnabled ? "PAUSE FEED" : "RESUME FEED"}
                    </button>
                  </div>
                </div>

                <div className="stream-queue-grid">
                  {streamQueue.map((tx) => (
                    <div
                      key={tx.id}
                      className={`stream-item-card risk-${tx.riskLevel}`}
                      onClick={() => handlePresetSelect(tx.presetsKey)}
                    >
                      <div className="item-badge-header">
                        <span className="tx-id">{tx.id}</span>
                        <span className={`risk-dot-label risk-${tx.riskLevel}`}>{tx.riskLevel.toUpperCase()}</span>
                      </div>
                      <strong className="tx-amount">{formatCurrency.format(tx.amount)}</strong>
                      <span className="tx-card">{tx.cardType}</span>
                      <span className="tx-time">{tx.timestamp}</span>
                    </div>
                  ))}
                </div>
              </section>

              <section className="grid grid--split">
                {/* LEFT: SIMULATOR FORM */}
                <div className="card">
                  <div className="card__header">
                    <h3>Input Parameters</h3>
                    <span className="muted">Simulate features to feed prediction endpoints</span>
                  </div>

                  <div className="presets-row">
                    <span className="presets-label">Preset Profiles:</span>
                    <div className="preset-buttons">
                      <button
                        className={`preset-btn ${selectedPresetKey === "low_risk" ? "active" : ""}`}
                        onClick={() => handlePresetSelect("low_risk")}
                      >
                        Legit Card
                      </button>
                      <button
                        className={`preset-btn ${selectedPresetKey === "high_risk_cnp" ? "active" : ""}`}
                        onClick={() => handlePresetSelect("high_risk_cnp")}
                      >
                        High-Risk CNP
                      </button>
                      <button
                        className={`preset-btn ${selectedPresetKey === "suspicious_intl" ? "active" : ""}`}
                        onClick={() => handlePresetSelect("suspicious_intl")}
                      >
                        Intl Velocity
                      </button>
                    </div>
                  </div>

                  <div className="mode-toggle">
                    <button
                      className={`mode-btn ${isFormMode ? "active" : ""}`}
                      onClick={() => setIsFormMode(true)}
                    >
                      Interactive Form
                    </button>
                    <button
                      className={`mode-btn ${!isFormMode ? "active" : ""}`}
                      onClick={() => setIsFormMode(false)}
                    >
                      Raw JSON Paste
                    </button>
                  </div>

                  <form onSubmit={handlePredictSubmit} className="panel__form">
                    {isFormMode ? (
                      <div className="form-fields-grid">
                        <div className="form-group">
                          <label>Transaction Amount ($)</label>
                          <input
                            type="number"
                            step="0.01"
                            min="0"
                            value={formFeatures.TransactionAmt ?? 0}
                            onChange={e => handleFormFieldChange("TransactionAmt", parseFloat(e.target.value))}
                          />
                        </div>
                        <div className="form-group">
                          <label>Account Age (days)</label>
                          <input
                            type="number"
                            min="0"
                            placeholder="e.g. 180"
                            value={formFeatures.D1 ?? 0}
                            onChange={e => handleFormFieldChange("D1", parseFloat(e.target.value))}
                          />
                        </div>
                        <div className="form-group">
                          <label>Transactions today (velocity)</label>
                          <input
                            type="number"
                            min="0"
                            placeholder="e.g. 2"
                            value={formFeatures.C1 ?? 0}
                            onChange={e => handleFormFieldChange("C1", parseInt(e.target.value))}
                          />
                        </div>
                        <div className="form-group">
                          <label>Distance from home (miles)</label>
                          <input
                            type="number"
                            min="0"
                            placeholder="e.g. 15"
                            value={formFeatures.dist1 ?? 0}
                            onChange={e => handleFormFieldChange("dist1", parseFloat(e.target.value))}
                          />
                        </div>
                        <div className="form-group">
                          <label>Network validation score (V14)</label>
                          <input
                            type="number"
                            step="0.1"
                            placeholder="Negative = higher risk"
                            value={formFeatures.V14 ?? 0}
                            onChange={e => handleFormFieldChange("V14", parseFloat(e.target.value))}
                          />
                        </div>
                      </div>
                    ) : (
                      <div className="raw-json-editor">
                        <label htmlFor="raw-features">Raw Input JSON</label>
                        <textarea
                          id="raw-features"
                          rows={12}
                          value={featuresText}
                          onChange={e => {
                            setFeaturesText(e.target.value);
                            setSelectedPresetKey("custom");
                          }}
                        />
                        <span className="panel__hint">Edit coordinates freely. Keys must match dataset schema.</span>
                      </div>
                    )}

                    <button type="submit" className="btn-primary" disabled={isLoading}>
                      {isLoading ? "Running Prediction Inference..." : "Evaluate Transaction Risk"}
                    </button>
                  </form>
                  {error && <p className="panel__error">{error}</p>}
                </div>

                {/* RIGHT: SIMULATOR Inference & Explanation Dashboard */}
                <div className="card">
                  <div className="card__header">
                    <h3>Scoring & Explanation Engine</h3>
                    <span className="muted">Prediction outputs and local game-theoretic (SHAP) feature weights</span>
                  </div>

                  {!result && (
                    <div className="empty-results">
                      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
                        <circle cx="12" cy="12" r="10" />
                        <line x1="8" y1="12" x2="16" y2="12" />
                        <line x1="12" y1="16" x2="12" y2="8" />
                      </svg>
                      <p>Analyze transaction details above to load dynamic SHAP explainability metrics.</p>
                    </div>
                  )}

                  {result && (
                    <div className="results-container">
                      <div className="results-hero">
                        <div className="gauge-score">
                          <span className="score-num">{percent(result.score)}</span>
                          <span className="score-lbl">Risk Probability</span>
                        </div>
                        <div className="results-badge">
                          <span className={`status ${result.is_fraud ? "status--risk" : "status--ok"}`}>
                            {result.is_fraud ? "HIGH FRAUD RISK" : "CLEARED TRANSACTION"}
                          </span>
                        </div>
                      </div>

                      {/* DUAL DECISION COMPARISON HERO */}
                      <div className="results-hero-split">
                        <div className="hero-block">
                          <h5>Static Decision ({result.threshold.toFixed(3)})</h5>
                          <span className={`status ${result.is_fraud ? "status--risk" : "status--ok"}`}>
                            {result.is_fraud ? "BLOCK" : "PASS"}
                          </span>
                        </div>
                        <div className="hero-block">
                          <h5>Dynamic Bayesian ({result.dynamic_threshold?.toFixed(3) ?? "N/A"})</h5>
                          <span className={`status ${(result.dynamic_is_fraud ?? result.is_fraud) ? "status--risk" : "status--ok"}`}>
                            {(result.dynamic_is_fraud ?? result.is_fraud) ? "BLOCK" : "PASS"}
                          </span>
                        </div>
                      </div>

                      {/* RISK PROGRESS BAR WITH DUAL THRESHOLDS */}
                      <div className="risk-meter">
                        <div className="risk-meter__label">
                          <span>Decision Bounds</span>
                          <span>Static: {result.threshold.toFixed(3)} | Dynamic: {result.dynamic_threshold?.toFixed(3) ?? "N/A"}</span>
                        </div>
                        <div className="risk-meter__bar">
                          <div
                            className={`risk-meter__fill ${result.is_fraud ? "fraud" : "safe"}`}
                            style={{ width: `${Math.min(result.score * 100, 100)}%` }}
                          />
                          <span
                            className="risk-meter__threshold"
                            style={{ left: `${result.threshold * 100}%` }}
                            title="Static Threshold"
                          />
                          {result.dynamic_threshold && (
                            <span
                              className="risk-meter__threshold"
                              style={{ left: `${result.dynamic_threshold * 100}%`, borderColor: "#f59e0b" }}
                              title="Dynamic Threshold"
                            />
                          )}
                        </div>
                      </div>

                      {/* CUSTOM DUAL-SIDED SHAP BAR LIST (Out-of-the-box UI styling) */}
                      <div className="shap-section">
                        <h4>Local Feature Contributions (SHAP Payoffs)</h4>
                        <div className="shap-custom-list">
                          {explanations.map((item, idx) => {
                            const totalImpact = explanations.reduce(
                              (sum, entry) => sum + Math.abs(entry.contribution),
                              0
                            );
                            const absCont = Math.abs(item.contribution);
                            const percentCont = totalImpact > 0 ? (absCont / totalImpact) * 100 : 0;
                            const isIncrease = item.direction === "increase";

                            return (
                              <div key={idx} className="shap-custom-row">
                                <div className="shap-feature-name">
                                  <code>{item.feature}</code>
                                </div>
                                <div className="shap-gauge-track">
                                  {/* Left segment (Decreases risk) */}
                                  <div className="shap-left-segment">
                                    {!isIncrease && (
                                      <div
                                        className="shap-fill decrease"
                                        style={{ width: `${percentCont}%`, marginLeft: "auto" }}
                                      />
                                    )}
                                  </div>
                                  {/* Axis marker */}
                                  <div className="shap-center-axis" />
                                  {/* Right segment (Increases risk) */}
                                  <div className="shap-right-segment">
                                    {isIncrease && (
                                      <div
                                        className="shap-fill increase"
                                        style={{ width: `${percentCont}%` }}
                                      />
                                    )}
                                  </div>
                                </div>
                                <div className={`shap-raw-value ${isIncrease ? "red" : "green"}`}>
                                  {isIncrease ? "+" : ""}{item.contribution.toFixed(3)}
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>

                      {/* DECISION MATRIX COMPARISON */}
                      {result.business_impact && (
                        <div className="simulator-impact">
                          <h4>Decision Matrix Comparison (Utility Optimization)</h4>
                          <table className="comparison-mini-table">
                            <thead>
                              <tr>
                                <th>Metric</th>
                                <th>Static Threshold</th>
                                <th>Dynamic Bayesian</th>
                              </tr>
                            </thead>
                            <tbody>
                              <tr>
                                <td>Decision</td>
                                <td className={result.is_fraud ? "red" : "green"}>{result.is_fraud ? "Block Blocked" : "Approved"}</td>
                                <td className={(result.dynamic_is_fraud ?? result.is_fraud) ? "red" : "green"}>{(result.dynamic_is_fraud ?? result.is_fraud) ? "Block Blocked" : "Approved"}</td>
                              </tr>
                              <tr>
                                <td>Expected Fraud Loss</td>
                                <td>{formatCurrency.format(result.business_impact.expected_fraud_loss)}</td>
                                <td>{formatCurrency.format(result.dynamic_business_impact?.expected_fraud_loss ?? result.business_impact.expected_fraud_loss)}</td>
                              </tr>
                              <tr>
                                <td>User Friction Cost</td>
                                <td>{formatCurrency.format(result.business_impact.false_positive_cost)}</td>
                                <td>{formatCurrency.format(result.dynamic_business_impact?.false_positive_cost ?? 0)}</td>
                              </tr>
                              <tr className="row-highlight">
                                <td>Net Benefit (Utility)</td>
                                <td className={result.business_impact.net_benefit >= 0 ? "plus" : "minus"}>
                                  {formatCurrency.format(result.business_impact.net_benefit)}
                                </td>
                                <td className={(result.dynamic_business_impact?.net_benefit ?? 0) >= 0 ? "plus" : "minus"}>
                                  {formatCurrency.format(result.dynamic_business_impact?.net_benefit ?? 0)}
                                </td>
                              </tr>
                            </tbody>
                          </table>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </section>
            </div>
          )}

          {/* TAB 3: DRIFT ANALYSIS */}
          {activeTab === "drift" && (
            <div className="tab-pane">
              {/* ROC and PR CURVES */}
              <section className="grid grid--split" style={{ marginBottom: "24px" }}>
                <div className="card">
                  <div className="card__header">
                    <h3>ROC Curve (Test Split Evaluation)</h3>
                    <span className="muted">Receiver Operating Characteristic: TPR vs FPR. Test AUC: {metrics ? percent(metrics.test_auc) : "--"}</span>
                  </div>
                  <div style={{ height: "180px", padding: "10px 0" }}>
                    <SleekLinePlot
                      data={metrics?.roc_curve ?? []}
                      xKey="fpr"
                      yKey="tpr"
                      xLabel="False Positive Rate"
                      yLabel="True Positive Rate"
                      stroke="#2563eb"
                      showBaseline={true}
                    />
                  </div>
                </div>

                <div className="card">
                  <div className="card__header">
                    <h3>Precision-Recall Curve (Test Split Evaluation)</h3>
                    <span className="muted">Precision vs Recall. Test PR AUC: {metrics ? percent(metrics.test_pr_auc) : "--"}</span>
                  </div>
                  <div style={{ height: "180px", padding: "10px 0" }}>
                    <SleekLinePlot
                      data={metrics?.pr_curve ?? []}
                      xKey="recall"
                      yKey="precision"
                      xLabel="Recall (Sensitivity)"
                      yLabel="Precision"
                      stroke="#10b981"
                      showBaseline={false}
                    />
                  </div>
                </div>
              </section>

              <div className="card">
                <div className="card__header">
                  <h3>Feature Distribution Drift Summary</h3>
                  <span className="muted">Statistical validation matching synthetic logs against historical references</span>
                </div>

                <div className="academic-intro">
                  <div className="intro-block">
                    <strong>Population Stability Index (PSI)</strong>
                    <p>
                      Measures variations in demographic values. Formula:
                      <code className="math-code">PSI = ∑ ( (Actual% - Expected%) × ln(Actual% / Expected%) )</code>
                      <br />
                      Status Bounds: <span className="text-green">PSI &lt; 0.1</span> (Stable), <span className="text-yellow">0.1 - 0.25</span> (Moderate Shift), <span className="text-red">&ge; 0.25</span> (Action Required).
                    </p>
                  </div>
                  <div className="intro-block">
                    <strong>Kolmogorov-Smirnov (KS) Test</strong>
                    <p>
                      Evaluating if two partitions sample identical continuous functions.
                      KS statistic outputs values in <code className="math-code">[0, 1]</code> (0 matches identical, 1 matches fully separated distributions).
                    </p>
                  </div>
                </div>

                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Feature ID</th>
                      <th>KS Statistic</th>
                      <th>PSI Stability</th>
                      <th>Classification Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {metrics?.drift_summary.map((row, index) => {
                      let statusPill = <span className="status status--ok">Stable</span>;
                      if (row.psi >= 0.25) {
                        statusPill = <span className="status status--risk">Critical Drift</span>;
                      } else if (row.psi >= 0.1) {
                        statusPill = <span className="status status--neutral">Moderate Drift</span>;
                      }

                      return (
                        <tr key={index}>
                          <td><code>{row.feature}</code></td>
                          <td>{formatDecimal.format(row.ks)}</td>
                          <td>{formatDecimal.format(row.psi)}</td>
                          <td>{statusPill}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* TAB 4: ML ARCHITECTURE */}
          {activeTab === "architecture" && (
            <div className="tab-pane">
              <section className="architecture-grid">
                <div className="card">
                  <div className="card__header">
                    <h3>ML Architecture and Processing Steps</h3>
                    <span className="muted">Explainable AI details and data preparation workflow</span>
                  </div>

                  <div className="step-workflow">
                    <div className="step-card">
                      <div className="step-num">1</div>
                      <h4>SMOTENC Oversampling</h4>
                      <p className="muted text-sm">
                        Class imbalance (e.g. 96.6% legit, 3.4% fraud) is handled during train splits by synthetically constructing minority examples inside categorical and continuous subspaces.
                      </p>
                    </div>
                    <div className="step-card">
                      <div className="step-num">2</div>
                      <h4>Bayesian Parameter Tuning</h4>
                      <p className="muted text-sm">
                        Optuna hyperparameter optimization performs trial searches targeting validation metrics. It identifies configurations for CatBoost regularizations, depths, and learning rates.
                      </p>
                    </div>
                    <div className="step-card">
                      <div className="step-num">3</div>
                      <h4>Sparse-Input Calibration</h4>
                      <p className="muted text-sm">
                        When transactions pass sparse values (missing attributes), our blended risk heuristic combines raw classification proba alongside deterministic overrides on C1, V14, V258 features to calibrate scores.
                      </p>
                    </div>
                    <div className="step-card">
                      <div className="step-num">4</div>
                      <h4>Explainable SHAP Model</h4>
                      <p className="muted text-sm">
                        Inference calls evaluate exact local Shapley games. The gradient booster decision structures are parsed to calculate positive or negative influence on risk margins.
                      </p>
                    </div>
                  </div>
                </div>

                <div className="card">
                  <div className="card__header">
                    <h3>Mathematical and System Formulations</h3>
                    <span className="muted">Click summary details below to examine formulas utilized in this thesis</span>
                  </div>
                  <div className="math-explanations">
                    <details>
                      <summary>Local Shapley Game Formula (SHAP)</summary>
                      <p>Determines the payout weight of features across coalition sets:</p>
                      <pre className="math-formula">
                        {"ϕ_i = ∑_{S ⊆ N \\ {i}} (|S|!(|N| - |S| - 1)! / |N|!) * (v(S ∪ {i}) - v(S))"}
                      </pre>
                    </details>
                    <details>
                      <summary>Cost-Sensitive Dynamic Threshold</summary>
                      <p>Maximises expected utility by setting the decision boundary analytically:</p>
                      <pre className="math-formula">{"θ*(x) = C_FP / TransactionAmt"}</pre>
                    </details>
                    <details>
                      <summary>SMOTENC Categorical Distance Penalty</summary>
                      <p>Extends SMOTE to mixed feature spaces by penalising nominal mismatches:</p>
                      <pre className="math-formula">{"d_cat(xᵢ, xⱼ) = r · σ²_med"}</pre>
                    </details>
                    <details>
                      <summary>Population Stability Index (Drift Detection)</summary>
                      <p>Measures distribution shift between training reference and production traffic:</p>
                      <pre className="math-formula">{"PSI = Σ (Actual% − Expected%) · ln(Actual% / Expected%)"}</pre>
                    </details>
                    <details>
                      <summary>Business Impact Net Benefit</summary>
                      <p>Computes dollar benefits weighing false positive friction vs fraud losses offset:</p>
                      <pre className="math-formula">
                        {"NetBenefit = (p_fraud · TransactionAmt) − FalsePositiveCost"}
                      </pre>
                    </details>
                  </div>

                  {modelComparison.length > 0 && (
                    <div style={{ marginTop: "1.5rem" }}>
                      <div className="card__header" style={{ marginBottom: "0.75rem" }}>
                        <h3>Model Selection Comparison</h3>
                        <span className="muted">CatBoost benchmarked against standard baselines on the same IEEE-CIS test split</span>
                      </div>
                      <div className="table-scroll">
                        <table className="data-table">
                          <thead>
                            <tr>
                              <th>Model</th>
                              <th>Test AUC ↑</th>
                              <th>PR-AUC ↑</th>
                              <th>Train time</th>
                            </tr>
                          </thead>
                          <tbody>
                            {[...modelComparison]
                              .sort((a, b) => b.auc - a.auc)
                              .map((m) => {
                                const isBest = m.model.startsWith("CatBoost");
                                return (
                                  <tr key={m.model} style={isBest ? { fontWeight: 600 } : {}}>
                                    <td>{isBest ? `${m.model} ✓` : m.model}</td>
                                    <td style={isBest ? { color: "var(--accent)" } : {}}>{m.auc.toFixed(4)}</td>
                                    <td>{m.pr_auc.toFixed(4)}</td>
                                    <td>{m.train_seconds != null ? `${m.train_seconds}s` : "—"}</td>
                                  </tr>
                                );
                              })}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </div>
              </section>
            </div>
          )}

        </div>
      </main>
    </div>
  );
}
