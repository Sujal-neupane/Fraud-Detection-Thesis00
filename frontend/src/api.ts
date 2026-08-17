export type PredictionResponse = {
  is_fraud: boolean;
  score: number;
  model_version: string;
  threshold: number;
  explanations: Array<{
    feature: string;
    contribution: number;
    direction: "increase" | "decrease";
  }>;
  business_impact?: {
    threshold: number;
    flagged: boolean;
    transaction_amount: number;
    expected_fraud_loss: number;
    false_positive_cost: number;
    net_benefit: number;
  };
  dynamic_threshold?: number;
  dynamic_is_fraud?: boolean;
  dynamic_business_impact?: {
    threshold: number;
    flagged: boolean;
    transaction_amount: number;
    expected_fraud_loss: number;
    false_positive_cost: number;
    net_benefit: number;
  };
};

export type MetricsResponse = {
  valid_auc: number;
  test_auc: number;
  valid_pr_auc: number;
  test_pr_auc: number;
  threshold: number;
  selected_precision: number;
  selected_recall: number;
  best_f1: number;
  business_impact: {
    threshold: number;
    flagged_per_10k: number;
    average_fraud_value: number;
    false_positive_cost: number;
    estimated_savings: number;
    estimated_friction: number;
    net_benefit: number;
  };
  drift_summary: Array<{
    feature: string;
    ks: number;
    psi: number;
  }>;
  roc_curve: Array<{ fpr: number; tpr: number }>;
  pr_curve: Array<{ precision: number; recall: number }>;
  global_importance: Array<{ feature: string; importance: number }>;
  rows: {
    train: number;
    valid: number;
    test: number;
  };
  fraud_rate: {
    train: number;
    valid: number;
    test: number;
  };
  model_version: string;
  input_path: string;
};

export type PresetsResponse = Record<string, Record<string, number>>;

export type ModelComparisonEntry = {
  model: string;
  auc: number;
  pr_auc: number;
  train_seconds: number | null;
};

export type ModelComparisonResponse = {
  models: ModelComparisonEntry[];
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export async function predict(
  features: Record<string, number>
): Promise<PredictionResponse> {
  const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features })
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const detail = typeof payload.detail === "string" ? payload.detail : "Request failed";
    throw new Error(detail);
  }

  return response.json();
}

export async function getPresets(): Promise<PresetsResponse> {
  const response = await fetch(`${API_BASE}/presets`);
  if (!response.ok) {
    throw new Error("Failed to load presets");
  }
  return response.json();
}

export async function getMetrics(): Promise<MetricsResponse> {
  const response = await fetch(`${API_BASE}/metrics`);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const detail = typeof payload.detail === "string" ? payload.detail : "Failed to load metrics";
    throw new Error(detail);
  }
  return response.json();
}

export async function getModelComparison(): Promise<ModelComparisonResponse> {
  const response = await fetch(`${API_BASE}/model-comparison`);
  if (!response.ok) return { models: [] };
  return response.json();
}
