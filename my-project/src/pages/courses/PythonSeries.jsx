import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { FaHome, FaAngleRight, FaBars } from "react-icons/fa";
import Sidebar from "../../components/Sidebar";
import SupportMeButton from "../../components/SupportMeButton";
import Comments from "../../components/Comments"; // เรียกใช้งานง่ายทุกหน้า

const lessons = [
  { id: "101", title: "Python Introduction", image: "/images/python-101.jpg", docLink: "#" },
  { id: "201", title: "Basic Data", image: "/images/python-201.jpg", docLink: "#" },
];

const PythonSeries = ({ theme }) => {
  const navigate = useNavigate();
  const [sidebarOpen, setSidebarOpen] = React.useState(false);

  return (
    <div className={`min-h-screen flex ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      {/* Hamburger Button สำหรับมือถือ */}
      <button
        className={`fixed top-4 left-4 z-50 p-2 rounded-md md:hidden ${
          theme === "dark" ? "bg-gray-700 text-white" : "bg-gray-200 text-black"
        }`}
        onClick={() => setSidebarOpen(true)}
      >
        <FaBars size={22} />
      </button>

      {/* Sidebar */}
      <Sidebar
        activeCourse="Python Series"
        theme={theme}
        sidebarOpen={sidebarOpen}
        setSidebarOpen={setSidebarOpen}
      />

      {/* Main Content */}
      <main className="flex-1 p-4 md:p-6 overflow-auto">
        <div className="max-w-3xl mx-auto">
          <div className="flex items-center gap-2 mb-4">
            <button
              className={`p-2 rounded-md ${
                theme === "dark" ? "bg-gray-700 text-white" : "bg-gray-300 text-black"
              }`}
              onClick={() => navigate("/")}
            >
              <FaHome size={18} />
            </button>
            <FaAngleRight className="text-gray-400" />
            <span className={`px-3 py-1 rounded-md ${
              theme === "dark" ? "bg-gray-700" : "bg-gray-300"
            }`}>
              Python Series
            </span>
          </div>

          <h1 className="text-3xl md:text-4xl font-bold mt-4">Python Series</h1>

          <div className={`p-3 rounded-md mt-4 ${
            theme === "dark" ? "bg-yellow-600 text-black" : "bg-yellow-400 text-black"
          }`}>
            ⚠ WARNING: เอกสารนี้อาจมีการเปลี่ยนแปลงตามเนื้อหาของหลักสูตร
          </div>

          {/* Responsive Table */}
          <div className="overflow-x-auto mt-4">
            <table className={`w-full border rounded-lg shadow-lg ${
              theme === "dark" ? "border-gray-700" : "border-gray-300"
            }`}>
              <thead className={`${theme === "dark" ? "bg-gray-800" : "bg-gray-200"}`}>
                <tr>
                  <th className="p-2">ตอน</th>
                  <th className="p-2">หัวข้อ</th>
                  <th className="p-2">วิดีโอ</th>
                  <th className="p-2">เอกสาร</th>
                </tr>
              </thead>
              <tbody>
                {lessons.map((lesson) => (
                  <tr key={lesson.id} className={`border-t ${
                    theme === "dark" ? "border-gray-700" : "border-gray-300"
                  }`}>
                    <td className="p-2">{lesson.id}</td>
                    <td className="p-2">{lesson.title}</td>
                    <td className="p-2">
                      <img src={lesson.image} className="w-20 rounded-lg cursor-pointer" />
                    </td>
                    <td className="p-2">
                      <a href={lesson.docLink} className="text-green-400 hover:underline">อ่าน</a>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* GitHub Comments Component ใช้ได้ทันที */}
          <Comments theme={theme} />
        </div>
      </main>

      <SupportMeButton />
    </div>
  );
};

export default PythonSeries;
