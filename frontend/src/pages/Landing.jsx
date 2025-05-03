import React from "react";
import Hero from "../components/Hero";
import NavBar from "../components/NavBar";
import Footer from "../components/Footer";
import { useNavigate } from "react-router";

export default function Landing() {
  const navigate = useNavigate();
  return (
    <div>
      <NavBar />
      <Hero onClick={() => navigate("/home")} />
      <Footer />
    </div>
  );
}
