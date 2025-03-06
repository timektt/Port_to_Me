import React, { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import CourseGrid from "./components/CourseGrid";
import PythonSeries from "./pages/courses/PythonSeries";
import SupportMeButton from "./components/SupportMeButton";
import Footer from "./components/Footer";

function App() {
  // ✅ โหลดธีมจาก localStorage อย่างปลอดภัย
  const [theme, setTheme] = useState(() => localStorage.getItem("theme") || "dark");

  // ✅ ใช้ useEffect เพื่ออัปเดตค่าใน <html>
  useEffect(() => {
    if (theme === "dark") {
      document.documentElement.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  }, [theme]);

  return (
    <Router>
      <div className={`min-h-screen flex flex-col ${theme === "dark" ? "bg-gray-900 text-white" : "bg-white text-gray-900"}`}>
        {/* ✅ ส่ง `theme` และ `setTheme` ไปที่ Navbar */}
        <Navbar theme={theme} setTheme={setTheme} />

        {/* ✅ Layout ของ Content */}
        <div className="flex-1">
          <Routes>
            {/* ✅ ส่ง `theme` และ `setTheme` ไปยัง PythonSeries */}
            <Route path="/" element={<CourseGrid theme={theme} />} />
            <Route path="/courses/python-series" element={<PythonSeries theme={theme} setTheme={setTheme} />} />
          </Routes>
        </div>

        {/* ✅ Footer และ Support Button */}
        <Footer />
        <div className="fixed bottom-16 right-4 z-50">
          <SupportMeButton />
        </div>
      </div>
    </Router>
  );
}

export default App;
