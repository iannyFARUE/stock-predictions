import axios from "axios";
export const getPrediction = async (tickers, date) => {
  const res = await axios.post("http://localhost:8000/api/v1/predict", {
    tickers,
    date,
  });
  return res.data;
};

export const getSentimentImpact = async (headline, ticker = "AAPL") => {
  const response = await axios.post(
    "http://localhost:8000/api/v1/predict-sentiment",
    { headline, ticker }
  );
  return response.data;
};
