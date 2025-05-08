// SentimentPage.jsx - Enhanced with detailed output, wider textarea, and result badges
import React, { useState } from "react";
import { getSentimentImpact } from "../api/stockService";
import NavBar from "../components/NavBar";
import Footer from "../components/Footer";

export default function SentimentPage() {
  const [sentimentInput, setSentimentInput] = useState("");
  const [sentimentImpact, setSentimentImpact] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    setLoading(true);
    try {
      const result = await getSentimentImpact(sentimentInput);
      setSentimentImpact(result);
    } catch (error) {
      console.error("Prediction error:", error);
    }
    setLoading(false);
  };

  const badgeStyle = (label) => {
    const styles = {
      NEUTRAL: "bg-yellow-100 text-yellow-800",
      POSITIVE: "bg-green-100 text-green-800",
      NEGATIVE: "bg-red-100 text-red-800",
      MINIMAL: "bg-gray-100 text-gray-800",
      MODERATE: "bg-blue-100 text-blue-800",
      STRONG: "bg-purple-100 text-purple-800",
      UP: "bg-green-200 text-green-900",
      DOWN: "bg-red-200 text-red-900",
    };
    return styles[label] || "bg-gray-100 text-gray-800";
  };

  return (
    <div className="flex flex-col min-h-screen bg-gray-50">
      <NavBar />
      <main className="flex-grow p-6 w-full max-w-3xl mx-auto">
        <h1 className="text-3xl font-bold mb-6 text-blue-900">
          Sentiment Impact Predictor
        </h1>

        <textarea
          value={sentimentInput}
          onChange={(e) => setSentimentInput(e.target.value)}
          className="w-full p-6 border rounded-lg mb-4 shadow text-lg"
          rows={12}
          placeholder="Paste a headline, article snippet, or market comment here..."
        />

        <button
          onClick={handleSubmit}
          className="bg-indigo-700 text-white px-6 py-2 rounded hover:scale-105 transition"
        >
          Analyze Sentiment
        </button>

        {loading && (
          <div className="mt-6 flex justify-center">
            <div className="w-12 h-12 border-4 border-indigo-600 border-t-transparent rounded-full animate-spin"></div>
          </div>
        )}

        {!loading && sentimentImpact && (
          <div className="mt-6 bg-white border rounded p-6 shadow text-gray-800">
            <h2 className="text-xl font-bold mb-4">
              📊 Sentiment Analysis Result
            </h2>
            {/* <p>
              <strong>TF-IDF Score:</strong> {sentimentImpact.tfidf_score}
            </p>
            <p>
              <strong>Financial Score:</strong>{" "}
              {sentimentImpact.financial_score}
            </p>
            <p>
              <strong>Adjusted Score:</strong> {sentimentImpact.adjusted_score}
            </p> */}
            <p>
              <strong>Category:</strong>
              <span
                className={`ml-2 px-2 py-1 rounded text-sm font-medium ${badgeStyle(
                  sentimentImpact.category
                )}`}
              >
                {sentimentImpact.category}
              </span>
            </p>
            <p>
              <strong>Impact:</strong>
              <span
                className={`ml-2 px-2 py-1 rounded text-sm font-medium ${badgeStyle(
                  sentimentImpact.impact
                )}`}
              >
                {sentimentImpact.impact}
              </span>
            </p>
            {/* <p>
              <strong>Predicted Return:</strong>{" "}
              {(sentimentImpact.predicted_return * 100).toFixed(2)}%
            </p> */}
            <p>
              <strong>Predicted Movement:</strong>
              <span
                className={`ml-2 px-2 py-1 rounded text-sm font-medium ${badgeStyle(
                  sentimentImpact.predicted_movement
                )}`}
              >
                {sentimentImpact.predicted_movement}
              </span>
            </p>

            {/* {sentimentImpact.terms?.length > 0 && (
              <div className="mt-4">
                <h3 className="font-semibold">Matched Terms:</h3>
                <ul className="list-disc pl-5">
                  {sentimentImpact.terms.map((term, idx) => (
                    <li key={idx}>{term}</li>
                  ))}
                </ul>
              </div>
            )} */}
          </div>
        )}
      </main>
      <Footer />
    </div>
  );
}
