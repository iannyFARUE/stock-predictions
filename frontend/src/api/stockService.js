import axios from "axios";
export const getPrediction = async (tickers, date) => {
  const res = await axios.post("http://localhost:8000/api/v1/predict", {
    tickers,
    date,
  });
  return res.data;
};
