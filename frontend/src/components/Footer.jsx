import React from "react";
export default function Footer() {
  return (
    <footer className="bg-blue-900 text-white text-center py-4">
      <p className="text-sm">
        &copy; {new Date().getFullYear()} SmartTrader. All rights reserved.
      </p>
    </footer>
  );
}
