import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import CourseGrid from "./components/CourseGrid";
import PythonSeries from "./pages/courses/PythonSeries"; // ✅ Import หน้าคอร์ส
import SupportMeButton from "./components/SupportMeButton";
import Footer from "./components/Footer";

function App() {
  return (
    <Router>
      <div className="bg-gray-900 min-h-screen text-white flex flex-col">
        <Navbar />
        <Routes>
          <Route path="/" element={<CourseGrid />} />
          <Route path="/courses/python-series" element={<PythonSeries />} /> 
          {/* ✅ เพิ่มเส้นทางไปยังหน้า Python Series */}
        </Routes>
        <SupportMeButton />
        <Footer />
      </div>
    </Router>
  );
}

export default App;
