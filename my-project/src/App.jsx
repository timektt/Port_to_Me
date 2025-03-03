import React from "react";
import Navbar from "./components/Navbar";
import VideoGrid from "./components/VideoGrid";
import SupportMeButton from "./components/SupportMeButton";  // ✅ Import ปุ่ม
import Footer from "./components/Footer";  // ✅ Import Footer

function App() {
  return (
    <div className="bg-gray-900 min-h-screen text-white">
      <Navbar />
      <VideoGrid />
      <SupportMeButton />  {/* ✅ แสดงปุ่ม Support Me */}
      <Footer />  {/* ✅ แสดง Footer */}
    </div>
  );
}

export default App;
