// Home.jsx - refreshed layout with clean Navbar, Footer, and Prediction logic
import React, { useState } from "react";
import { getPrediction } from "../api/stockService";
import PredictionCard from "../components/PredictionCard";
import NavBar from "../components/NavBar";
import Footer from "../components/Footer";

const allTickers = ["AAPL", "GOOGL", "AMZN", "MSFT", "NVDA"];

export default function Home() {
  const [prediction, setPrediction] = useState(null);
  const [selected, setSelected] = useState(allTickers.slice(0, 3));
  const [chartData, setChartData] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleTickerChange = (ticker) => {
    setSelected((prev) =>
      prev.includes(ticker)
        ? prev.filter((t) => t !== ticker)
        : [...prev, ticker]
    );
  };

  const handlePredict = async () => {
    setLoading(true);
    const result = await getPrediction(
      selected,
      new Date().toISOString().split("T")[0]
    );
    setPrediction(result);

    const dummyChart = result.map((r) => ({
      date: new Date().toISOString().split("T")[0],
      price: r.current_price,
      ticker: r.ticker,
    }));
    setChartData(dummyChart);
    setLoading(false);
  };

  return (
    <div className="flex flex-col min-h-screen bg-gray-50">
      <NavBar />

      <main className="flex-grow p-6 max-w-5xl mx-auto">
        <h1 className="text-2xl font-bold mb-4">Stock Analysis Dashboard</h1>

        <div className="mb-4">
          <p className="mb-2 font-medium">Select Stocks to Analyze:</p>
          <div className="flex flex-wrap gap-2">
            {allTickers.map((ticker) => (
              <button
                key={ticker}
                onClick={() => handleTickerChange(ticker)}
                className={`px-4 py-2 rounded border ${
                  selected.includes(ticker)
                    ? "bg-green-600 text-white"
                    : "bg-white text-gray-800"
                }`}
              >
                {ticker}
              </button>
            ))}
          </div>
        </div>

        <button
          onClick={handlePredict}
          className="bg-blue-600 text-white px-6 py-2 rounded shadow hover:scale-105 transition mb-6"
        >
          Predict Best Stock
        </button>

        {loading && (
          <div className="flex justify-center items-center py-8">
            <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
          </div>
        )}

        {!loading && prediction && (
          <PredictionCard
            prediction={prediction}
            chartData={chartData}
            selected={selected}
          />
        )}
      </main>

      <Footer />
    </div>
  );
}
