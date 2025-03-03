import React from "react";
import Navbar from "./components/Navbar";
import VideoGrid from "./components/VideoGrid";
import SupportMeButton from "./components/SupportMeButton";
import Footer from "./components/Footer"; // 🔹 Import Footer

function App() {
  return (
    <div className="bg-gray-900 min-h-screen text-white flex flex-col">
      <Navbar />
      <VideoGrid />
      <SupportMeButton />
      <Footer /> {/* 🔹 เพิ่ม Footer ที่ด้านล่าง */}
    </div>
  );
}

export default App;
