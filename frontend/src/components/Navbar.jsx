import React from "react";
import { NavLink } from "react-router";

export default function NavBar() {
  return (
    <nav className="bg-blue-900 text-white px-8 py-4 flex justify-between items-center">
      <h1 className="text-xl font-bold">
        <NavLink to="/">SmartTrader</NavLink>
      </h1>
      <div className="space-x-4">
        <NavLink to="/" className="hover:underline">
          Home
        </NavLink>
        <NavLink to="/home" className="hover:underline">
          Technical Analysis
        </NavLink>
        <NavLink to="/sentiment" className="hover:underline">
          Sentiment Analysis
        </NavLink>
      </div>
    </nav>
  );
}
