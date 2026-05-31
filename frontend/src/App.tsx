import { useState } from "react";
import "./App.css";

type Prediction = {
  label: string;
  confidence: number;
};

type PredictResponse = {
  predicted_label: string;
  confidence: number;
  top_predictions: Prediction[];
};

const App = () => {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>("");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];

    setResult(null);
    setError("");

    if (!selectedFile) return;

    if (!selectedFile.type.startsWith("image/")) {
      setFile(null);
      setPreviewUrl("");
      setError("Please upload an image file, such as JPG, PNG, or WebP.");
      return;
    }

    setFile(selectedFile);
    setPreviewUrl(URL.createObjectURL(selectedFile));
  };

  const handlePredict = async () => {
    if (!file) {
      setError("Please select an image first.");
      return;
    }

    setIsLoading(true);
    setError("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || "Prediction failed.");
      }

      const data = (await response.json()) as PredictResponse;
      setResult(data);
    } catch (err) {
      if (err instanceof TypeError && err.message === "Failed to fetch") {
        setError(
          "Cannot connect to backend API. Please check FastAPI is running on http://localhost:8000.",
        );
      } else {
        setError(err instanceof Error ? err.message : "Unexpected error.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="app">
      <section className="panel">
        <h1>Image Classifier Demo</h1>

        <input type="file" accept="image/*" onChange={handleFileChange} />

        {previewUrl && (
          <img className="preview" src={previewUrl} alt="Selected preview" />
        )}

        <button onClick={handlePredict} disabled={!file || isLoading}>
          {isLoading ? "Predicting..." : "Predict"}
        </button>

        {error && <p className="error">{error}</p>}

        {result && (
          <section className="result">
            <h2>{result.predicted_label}</h2>
            <p>Confidence: {(result.confidence * 100).toFixed(2)}%</p>

            <h3>Top predictions</h3>
            <ul>
              {result.top_predictions.map((prediction) => (
                <li key={prediction.label}>
                  {prediction.label}: {(prediction.confidence * 100).toFixed(2)}
                  %
                </li>
              ))}
            </ul>
          </section>
        )}
      </section>
    </main>
  );
};

export default App;
