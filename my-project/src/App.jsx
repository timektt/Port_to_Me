import React, { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/common/Navbar";
import CourseGrid from "./home/CourseGrid";
import PythonSeries from "./pages/courses/PythonSeries";
import NodeSeries from "./pages/courses/NodejsSeries"; // ✅ Import Node.js Series
import RestfulApiGraphQLSeries from "./pages/courses/RestfulApiGraphQLSeries"; // ✅ Import RESTful API & GraphQL Series
import ReactJsSeries from "./pages/courses/ReactJsSeries"; // ✅ Import React.js Series
import WebDevSeries from "./pages/courses/WebDevSeries"; // ✅ Import Web Development Series
import BasicProgrammingSeries from "./pages/courses/BasicProgrammingSeries"; // ✅ Import Basic Programming Series
import AllCourses from "./pages/courses/AllCourses"; // ✅ Import AllCourses.jsx
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
            {/* ✅ หน้าแรก - แสดงคอร์สล่าสุด */}
            <Route path="/" element={<CourseGrid theme={theme} />} />

            {/* ✅ หน้าคอร์สทั้งหมด */}
            <Route path="/courses" element={<AllCourses theme={theme} setTheme={setTheme} />} />

            {/* ✅ หน้ารายละเอียดของคอร์ส Python Series */}
            <Route path="/courses/python-series" element={<PythonSeries theme={theme} setTheme={setTheme} />} />

            {/* ✅ หน้ารายละเอียดของคอร์ส Node.js Series */}
            <Route path="/courses/nodejs-series" element={<NodeSeries theme={theme} setTheme={setTheme} />} /> 

            {/* ✅ หน้ารายละเอียดของคอร์ส RESTful API & GraphQL Series */}
            <Route path="/courses/restful-api-graphql-series" element={<RestfulApiGraphQLSeries theme={theme} setTheme={setTheme} />} />

            {/* ✅ หน้ารายละเอียดของคอร์ส React.js Series */}
            <Route path="/courses/reactjs-series" element={<ReactJsSeries theme={theme} setTheme={setTheme} />} />

            {/* ✅ หน้ารายละเอียดของคอร์ส Web Development Series */}
            <Route path="/courses/web-development" element={<WebDevSeries theme={theme} setTheme={setTheme} />} />

            {/* ✅ หน้ารายละเอียดของคอร์ส Basic Programming Series */}
            <Route path="/courses/basic-programming" element={<BasicProgrammingSeries theme={theme} setTheme={setTheme} />} /> {/* ✅ เพิ่ม Route ของ Basic Programming Series */}
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
