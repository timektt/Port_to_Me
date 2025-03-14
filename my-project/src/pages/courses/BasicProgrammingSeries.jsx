import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../../components/common/Navbar";
import BasicProgrammingSidebar from "../../components/common/sidebar/BasicProgrammingSidebar"; // ✅ Import Sidebar ของ Basic Programming
import SupportMeButton from "../../support/SupportMeButton";
import Comments from "../../components/common/Comments";
import BasicProgrammingMobileMenu from "../../components/common/sidebar/MobileMenus/BasicProgrammingMobileMenu"; // ✅ Import Mobile Menu ของ Basic Programming
import Breadcrumb from "../../components/common/Breadcrumb"; // ✅ ใช้ Breadcrumb

const lessons = [
  { id: "101", title: "Introduction to Programming", image: "/basic1.png", docLink: "#", videoLink: "#" },
  { id: "201", title: "Variables & Data Types", image: "/basic2.jpg", docLink: "#", videoLink: "#" },
  { id: "202", title: "Control Flow & Loops", image: "/basic3.jpg", docLink: "#", videoLink: "#" },
];

const BasicProgrammingSeries = ({ theme, setTheme }) => {
  const navigate = useNavigate();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth >= 768) {
        setSidebarOpen(false);
        setMobileMenuOpen(false);
      }
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (
    <div className={`min-h-screen flex flex-col ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      {/* ✅ Navbar */}
      <div className="fixed top-0 left-0 w-full z-50">
        <Navbar theme={theme} setTheme={setTheme} onMenuToggle={() => setMobileMenuOpen(!mobileMenuOpen)} />
      </div>

      {/* ✅ Sidebar */}
      <div className="hidden md:block fixed left-0 top-16 h-[calc(100vh-4rem)] w-64 z-40">
        <BasicProgrammingSidebar theme={theme} sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen} />
      </div>

      {/* ✅ Mobile Sidebar */}
      {mobileMenuOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50">
          <BasicProgrammingMobileMenu onClose={() => setMobileMenuOpen(false)} theme={theme} setTheme={setTheme} />
        </div>
      )}

      {/* ✅ Main Content */}
      <main className="flex-1 md:ml-64 p-4 md:p-6 mt-16 relative z-10">
        <div className="max-w-5xl mx-auto">
          {/* ✅ Breadcrumb Navigation */}
          <Breadcrumb courseName="Basic Programming" theme={theme} />

          <h1 className="text-3xl md:text-4xl font-bold mt-4">Basic Programming Series</h1>

          {/* ✅ Warning Box */}
          <div className={`p-4 mt-4 rounded-md shadow-md flex flex-col gap-2 ${theme === "dark" ? "bg-yellow-700 text-white" : "bg-yellow-300 text-black"}`}>
            <strong className="text-lg flex items-center gap-2">⚠ WARNING</strong>
            <p>เอกสารฉบับนี้ยังอยู่ในระหว่างการทำ Series ของ Basic Programming...</p>
          </div>

          {/* ✅ Table Section */}
          <div className="overflow-x-auto mt-6">
            <table className={`w-full border rounded-lg shadow-lg ${theme === "dark" ? "border-gray-700" : "border-gray-300"}`}>
              <thead className={`${theme === "dark" ? "bg-gray-800 text-white" : "bg-gray-300 text-black"} text-lg`}>
                <tr>
                  <th className="p-4 border-b-2 w-1/6">ตอน</th>
                  <th className="p-4 border-b-2 w-1/3">หัวข้อ</th>
                  <th className="p-4 border-b-2 w-1/3">วิดีโอ</th>
                  <th className="p-4 border-b-2 w-1/6">เอกสาร</th>
                </tr>
              </thead>
              <tbody>
                {lessons.map((lesson, index) => (
                  <tr key={lesson.id} className={`${index % 2 === 0 ? (theme === "dark" ? "bg-gray-700" : "bg-gray-100") : ""} hover:bg-gray-500 transition duration-200`}>
                    <td className="p-4 text-center border-b text-lg font-semibold">{lesson.id}</td>
                    <td className="p-4 border-b text-lg">{lesson.title}</td>
                    <td className="p-4 border-b text-center">
                      <a href={lesson.videoLink} target="_blank" rel="noopener noreferrer">
                        <img src={lesson.image} className="w-80 h-60 mx-auto rounded-lg shadow-lg cursor-pointer transition-transform transform hover:scale-105 hover:shadow-xl object-cover" alt={lesson.title} />
                        <span className="block mt-2 text-green-400 hover:underline">ดู video</span>
                      </a>
                    </td>
                    <td className="p-4 border-b text-center">
                      <a href={lesson.docLink} target="_blank" rel="noopener noreferrer" className="text-green-400 hover:underline">อ่าน</a>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* ✅ Comments Section */}
          <Comments theme={theme} />
        </div>
      </main>

      <SupportMeButton />
    </div>
  );
};

export default BasicProgrammingSeries;
