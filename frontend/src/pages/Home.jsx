// Home.jsx - improved dashboard with selection and visuals
import React, { useState } from "react";
import { getPrediction } from "../api/stockService";
import PredictionCard from "../components/PredictionCard";
import StockChart from "../components/StockChart";
import NavBar from "../components/NavBar";
import Footer from "../components/Footer";

const allTickers = ["AAPL", "GOOGL", "MSFT", "TSLA", "META", "NVDA"];

export default function Home() {
  const [prediction, setPrediction] = useState(null);
  const [selected, setSelected] = useState(allTickers.slice(0, 3));
  const [chartData, setChartData] = useState([]);

  const handleTickerChange = (ticker) => {
    setSelected((prev) =>
      prev.includes(ticker)
        ? prev.filter((t) => t !== ticker)
        : [...prev, ticker]
    );
  };

  const handlePredict = async () => {
    const result = await getPrediction(
      selected,
      new Date().toISOString().split("T")[0]
    );
    setPrediction(result);

    // Dummy chart data simulation
    const dummyChart = result.rankings.map((r) => ({
      date: new Date().toISOString().split("T")[0],
      price: 100 + Math.random() * 10,
      ticker: r.ticker,
    }));
    setChartData(dummyChart);
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

        {prediction && (
          <>
            <PredictionCard prediction={prediction} />
            {selected.map((ticker) => (
              <StockChart
                key={ticker}
                ticker={ticker}
                data={chartData.filter((d) => d.ticker === ticker)}
              />
            ))}
          </>
        )}
      </main>
      <Footer />
    </div>
  );
}
