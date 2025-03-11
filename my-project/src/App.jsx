import React, { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/common/Navbar";
import CourseGrid from "./home/CourseGrid";
import PythonSeries from "./pages/courses/PythonSeries";
import NodeSeries from "./pages/courses/NodejsSeries";
import RestfulApiGraphQLSeries from "./pages/courses/RestfulApiGraphQLSeries";
import ReactJsSeries from "./pages/courses/ReactJsSeries";
import WebDevSeries from "./pages/courses/WebDevSeries";
import BasicProgrammingSeries from "./pages/courses/BasicProgrammingSeries";
import AllCourses from "./pages/courses/AllCourses";
import SupportMeButton from "./support/SupportMeButton";
import Footer from "./components/common/Footer";
import BasicProgrammingMobileMenu from "./components/common/sidebar/MobileMenus/BasicProgrammingMobileMenu"; // ✅ Import Mobile Menu

// ✅ Import ไฟล์หัวข้อย่อยของแต่ละคอร์ส
import PythonIntro from "./pages/courses/topics/python/PythonIntro";
import PythonVariables from "./pages/courses/topics/python/PythonVariables";
import PythonControlStructure from "./pages/courses/topics/python/PythonControlStructure";
import PythonInputFunction from "./pages/courses/topics/python/PythonInputFunction";
import PythonLeetcode from "./pages/courses/topics/python/PythonLeetcode";

import NodeIntro from "./pages/courses/topics/nodejs/NodeIntro";
import NodeAsync from "./pages/courses/topics/nodejs/NodeAsync";
import NodeRestApi from "./pages/courses/topics/nodejs/NodeRestApi";

import ReactIntro from "./pages/courses/topics/reactjs/ReactIntro";
import ReactComponents from "./pages/courses/topics/reactjs/ReactComponents";
import ReactState from "./pages/courses/topics/reactjs/ReactState";
import ReactHooks from "./pages/courses/topics/reactjs/ReactHooks";

import WebHtmlBasics from "./pages/courses/topics/webdev/WebHtmlBasics";
import WebCssBasics from "./pages/courses/topics/webdev/WebCssBasics";
import WebJsBasics from "./pages/courses/topics/webdev/WebJsBasics";

import BasicProgrammingIntro from "./pages/courses/topics/basic-programming/BasicProgrammingIntro";
import Algorithms from "./pages/courses/topics/basic-programming/Algorithms";
import DataStructures from "./pages/courses/topics/basic-programming/DataStructures";

function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem("theme") || "dark");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false); // ✅ State ควบคุมเมนู Mobile

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    localStorage.setItem("theme", theme);
  }, [theme]);

  return (
    <Router>
      <div className={`min-h-screen flex flex-col ${theme === "dark" ? "bg-gray-900 text-white" : "bg-white text-gray-900"}`}>
        {/* ✅ ส่ง `onMenuToggle` ไปที่ Navbar เพื่อให้กดเปิดเมนู Mobile ได้ */}
        <Navbar theme={theme} setTheme={setTheme} onMenuToggle={() => setMobileMenuOpen(true)} />

        {/* ✅ แสดง Mobile Menu เมื่อ `mobileMenuOpen === true` */}
        {mobileMenuOpen && (
          <BasicProgrammingMobileMenu onClose={() => setMobileMenuOpen(false)} theme={theme} setTheme={setTheme} />
        )}

        <div className="flex-1">
          <Routes>
            {/* ✅ หน้าแรก */}
            <Route path="/" element={<CourseGrid theme={theme} />} />

            {/* ✅ หน้าคอร์สทั้งหมด */}
            <Route path="/courses" element={<AllCourses theme={theme} setTheme={setTheme} />} />

            {/* ✅ หน้าหลักของแต่ละคอร์ส */}
            <Route path="/courses/python-series" element={<PythonSeries theme={theme} setTheme={setTheme} />} />
            <Route path="/courses/nodejs-series" element={<NodeSeries theme={theme} setTheme={setTheme} />} />
            <Route path="/courses/reactjs-series" element={<ReactJsSeries theme={theme} setTheme={setTheme} />} />
            <Route path="/courses/web-development" element={<WebDevSeries theme={theme} setTheme={setTheme} />} />
            <Route path="/courses/basic-programming" element={<BasicProgrammingSeries theme={theme} setTheme={setTheme} />} />
            <Route path="/courses/restful-api-graphql-series" element={<RestfulApiGraphQLSeries theme={theme} setTheme={setTheme} />} />

            {/* ✅ Python Subtopics */}
            <Route path="/courses/python-series/intro" element={<PythonIntro />} />
            <Route path="/courses/python-series/variables" element={<PythonVariables />} />
            <Route path="/courses/python-series/control-structure" element={<PythonControlStructure />} />
            <Route path="/courses/python-series/input-function" element={<PythonInputFunction />} />
            <Route path="/courses/python-series/leetcode" element={<PythonLeetcode />} />

            {/* ✅ Node.js Subtopics */}
            <Route path="/courses/nodejs-series/intro" element={<NodeIntro />} />
            <Route path="/courses/nodejs-series/async" element={<NodeAsync />} />
            <Route path="/courses/nodejs-series/rest-api" element={<NodeRestApi />} />

            {/* ✅ React.js Subtopics */}
            <Route path="/courses/reactjs-series/intro" element={<ReactIntro />} />
            <Route path="/courses/reactjs-series/components" element={<ReactComponents />} />
            <Route path="/courses/reactjs-series/state" element={<ReactState />} />
            <Route path="/courses/reactjs-series/hooks" element={<ReactHooks />} />

            {/* ✅ Web Development Subtopics */}
            <Route path="/courses/web-development/html-basics" element={<WebHtmlBasics />} />
            <Route path="/courses/web-development/css-basics" element={<WebCssBasics />} />
            <Route path="/courses/web-development/javascript-basics" element={<WebJsBasics />} />

            {/* ✅ Basic Programming Subtopics */}
            <Route path="/courses/basic-programming/intro" element={<BasicProgrammingIntro />} />
            <Route path="/courses/basic-programming/algorithms" element={<Algorithms />} />
            <Route path="/courses/basic-programming/data-structures" element={<DataStructures />} />
          </Routes>
        </div>

        <Footer />
        <div className="fixed bottom-16 right-4 z-50">
          <SupportMeButton />
        </div>
      </div>
    </Router>
  );
}

export default App;
