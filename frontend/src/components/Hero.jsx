import React from "react";
import heroImg from "../assets/stock-t1.jpg"; // Ensure this PNG image is present

export default function Hero({ onClick }) {
  return (
    <section className="bg-gradient-to-br from-blue-900 to-indigo-800 text-white min-h-[90vh] flex flex-col md:flex-row items-center justify-between px-6 md:px-20 py-10">
      <div className="md:w-1/2 mb-10 md:mb-0">
        <h1 className="text-4xl md:text-5xl font-bold mb-4">
          Predict Tomorrow’s Winners Today
        </h1>
        <p className="text-md md:text-lg mb-6 text-white/90">
          Our AI-driven engine analyzes market sentiment and technical
          indicators to recommend the best stock to buy now.
        </p>
        <button
          onClick={onClick}
          className="bg-white text-blue-800 px-6 py-3 rounded-xl font-semibold hover:scale-105 transition-all"
        >
          Get Started
        </button>
      </div>
      <div className="md:w-1/2 flex justify-center">
        <img
          src={heroImg}
          alt="Stock Market Hero"
          className="w-full max-w-2xl rounded-lg shadow-lg"
        />
      </div>
    </section>
  );
}
