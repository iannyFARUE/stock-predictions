import axios from "axios";
export const getPrediction = async (tickers, date) => {
  const res = await axios.post("/api/v1/predict", { tickers, date });
  return res.data;
};
