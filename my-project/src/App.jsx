import React from "react";
import Navbar from "./components/Navbar";
import VideoGrid from "./components/VideoGrid";

function App() {
  return (
    <div className="bg-gray-900 min-h-screen text-white">
      <Navbar />
      <VideoGrid />
    </div>
  );
}

export default App;
