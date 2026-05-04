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
