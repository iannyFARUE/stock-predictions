import React from "react";
export default function Footer() {
  return (
    <footer className="bg-blue-900 text-white text-center py-4">
      <p className="text-sm">
        &copy; {new Date().getFullYear()} SmartTrader, made with &#x1F49E; by
        Ian Madhara, Simbaremuteuro Chin'ombe, Dickson Kachepa and Willard
        Chipangura. All rights reserved.
      </p>
    </footer>
  );
}
