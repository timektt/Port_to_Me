import React, { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/common/Navbar";
import CourseGrid from "./home/CourseGrid";
import PythonSeries from "./pages/courses/PythonSeries";
import SupportMeButton from "./support/SupportMeButton";
import Footer from "./components/common/Footer";


function App() {
  // ✅ โหลดธีมจาก localStorage อย่างปลอดภัย
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem("theme") || "dark";
  });

  // ✅ ใช้ useEffect เพื่ออัปเดตค่าใน <html> และ localStorage
  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    localStorage.setItem("theme", theme);
    console.log("🌓 Theme changed to:", theme);
  }, [theme]);

  // ✅ ฟัง event `storage` เพื่อให้ธีมเปลี่ยนในทุกแท็บที่เปิดอยู่
  useEffect(() => {
    const syncTheme = (event) => {
      if (event.key === "theme") {
        setTheme(event.newValue);
      }
    };
    window.addEventListener("storage", syncTheme);
    return () => window.removeEventListener("storage", syncTheme);
  }, []);

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
