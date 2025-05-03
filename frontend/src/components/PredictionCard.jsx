import React from "react";

export default function PredictionCard({ prediction }) {
  return (
    <div className="mt-6 p-4 border rounded shadow">
      <h2 className="text-lg font-semibold">
        Recommended Stock: {prediction.recommended_ticker}
      </h2>
      <p>
        Expected Return (10d):{" "}
        {Math.round(prediction.expected_return * 10000) / 100}%
      </p>
      <h3 className="mt-2 font-medium">Rankings:</h3>
      <ul>
        {prediction.rankings.map((r, idx) => (
          <li key={idx}>
            {r.ticker}: {(r.predicted_return * 100).toFixed(2)}%
          </li>
        ))}
      </ul>
    </div>
  );
}
