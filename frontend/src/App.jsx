// frontend/src/App.js
import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router";
import Landing from "./pages/Landing";
import Home from "./pages/Home";
import SentimentPage from "./pages/Sentiment";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/home" element={<Home />} />
        <Route path="/sentiment" element={<SentimentPage />} />
      </Routes>
    </Router>
  );
}

export default App;
