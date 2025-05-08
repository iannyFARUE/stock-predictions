// PredictionCard.jsx - updated with badge styling for recommendations
import React from "react";
import StockChart from "./StockChart";

export default function PredictionCard({ prediction, chartData, selected }) {
  if (!Array.isArray(prediction) || prediction.length === 0) {
    return (
      <div className="mt-6 p-4 border rounded shadow">
        No predictions available.
      </div>
    );
  }

  const filtered = prediction.filter((p) => selected.includes(p.ticker));
  const sorted = [...filtered].sort(
    (a, b) => b.predicted_return - a.predicted_return
  );
  const top = sorted[0];

  const getBadgeColor = (recommendation) => {
    switch (recommendation) {
      case "Buy":
        return "bg-green-100 text-green-800";
      case "Sell":
        return "bg-red-100 text-red-800";
      case "Do not Enter":
        return "bg-yellow-100 text-yellow-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

  return (
    <div className="mt-6 p-6 border rounded-lg shadow bg-white">
      <h2 className="text-2xl font-bold text-blue-800 mb-2">
        Top Recommendation: {top?.ticker || "N/A"}
      </h2>
      <p className="text-gray-700 mb-4">
        📈 Expected Return (30d): {top?.predicted_return_percent?.toFixed(2)}%{" "}
        <br />
        💰 Current Price: ${top?.current_price?.toFixed(2)} <br />
        📅 Date: {top?.prediction_date} <br />✅ Recommendation:
        <span
          className={`ml-2 px-2 py-1 rounded text-sm font-medium ${getBadgeColor(
            top?.recommendation
          )}`}
        >
          {top?.recommendation}
        </span>
      </p>

      <h3 className="text-lg font-semibold mb-2">Selected Predictions:</h3>
      <table className="w-full text-sm text-left text-gray-600 mb-6">
        <thead className="text-xs uppercase bg-gray-100">
          <tr>
            <th className="px-2 py-2">Ticker</th>
            <th className="px-2 py-2">Return %</th>
            <th className="px-2 py-2">Price</th>
            <th className="px-2 py-2">Recommendation</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((stock, idx) => (
            <tr key={idx} className="border-b">
              <td className="px-2 py-2 font-medium">{stock.ticker}</td>
              <td className="px-2 py-2">
                {stock.predicted_return_percent.toFixed(2)}%
              </td>
              <td className="px-2 py-2">${stock.current_price.toFixed(2)}</td>
              <td className="px-2 py-2">
                <span
                  className={`px-2 py-1 rounded text-xs font-semibold ${getBadgeColor(
                    stock.recommendation
                  )}`}
                >
                  {stock.recommendation}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3 className="text-lg font-semibold mb-2">Stock Charts:</h3>
      {sorted.map((stock) => (
        <StockChart
          key={stock.ticker}
          ticker={stock.ticker}
          data={chartData.filter((d) => d.ticker === stock.ticker)}
        />
      ))}
    </div>
  );
}
