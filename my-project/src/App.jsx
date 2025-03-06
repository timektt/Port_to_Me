import React, { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import CourseGrid from "./components/CourseGrid";
import PythonSeries from "./pages/courses/PythonSeries";
import SupportMeButton from "./components/SupportMeButton";
import Footer from "./components/Footer";

function App() {
  // ✅ โหลดธีมจาก localStorage
  const [theme, setTheme] = useState(localStorage.getItem("theme") || "dark");

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
        <Navbar theme={theme} setTheme={setTheme} />
        <Routes>
          <Route path="/" element={<CourseGrid theme={theme} />} />
          <Route path="/courses/python-series" element={<PythonSeries theme={theme} />} />
        </Routes>
        <SupportMeButton />
        <Footer />
      </div>
    </Router>
  );
}

export default App;
