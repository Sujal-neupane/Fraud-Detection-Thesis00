import { FormEvent, useMemo, useState } from "react";

import { predict, PredictionResponse } from "./api";

const DEFAULT_FEATURES = {
  feature_1: 0.12,
  feature_2: 5.4
};

function App() {
  const [featuresText, setFeaturesText] = useState(
    JSON.stringify(DEFAULT_FEATURES, null, 2)
  );
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const exampleSchema = useMemo(
    () => "Expected JSON: { \"feature_1\": 0.12, \"feature_2\": 5.4 }",
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

  return (
    <div className="app">
      <header className="app__header">
        <h1>Fraud Detection Demo</h1>
        <p>
          Paste a JSON feature map to test the prediction API. This will connect
          to the backend once the model pipeline is implemented.
        </p>
      </header>

      <section className="panel">
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
              <p>
                <strong>Fraud:</strong> {result.is_fraud ? "Yes" : "No"}
              </p>
              <p>
                <strong>Score:</strong> {result.score.toFixed(4)}
              </p>
              <p>
                <strong>Model:</strong> {result.model_version}
              </p>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

export default App;
