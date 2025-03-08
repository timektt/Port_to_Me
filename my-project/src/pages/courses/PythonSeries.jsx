import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { FaHome, FaAngleRight } from "react-icons/fa"; // ✅ ใช้ไอคอนบ้าน
import Navbar from "../../components/Navbar";
import Sidebar from "../../components/Sidebar";
import SupportMeButton from "../../components/SupportMeButton";
import Comments from "../../components/Comments";
import Footer from "../../components/Footer";
import MobileMenu from "../../components/MobileMenu"; // ✅ เพิ่ม MobileMenu

const lessons = [
  { id: "101", title: "Python Introduction", image: "/images/python-101.jpg", docLink: "#" },
  { id: "201", title: "Basic Data", image: "/images/python-201.jpg", docLink: "#" },
];

const PythonSeries = ({ theme, setTheme }) => {
  const navigate = useNavigate();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false); // ✅ เพิ่ม state ควบคุม Mobile Sidebar

  // ✅ ปิด Sidebar อัตโนมัติเมื่อขยายหน้าจอกลับเป็น Desktop
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth >= 768) {
        setSidebarOpen(false);
      }
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (
    <div className={`min-h-screen flex flex-col ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      {/* ✅ Sticky Navbar */}
      <div className="fixed top-0 left-0 w-full z-50">
        <Navbar theme={theme} setTheme={setTheme} onMenuToggle={() => setMobileMenuOpen(!mobileMenuOpen)} />
      </div>

      {/* ✅ Sidebar (Fixed) - แสดงเฉพาะจอใหญ่ */}
      <div className="hidden md:block fixed left-0 top-16 h-[calc(100vh-4rem)] w-64 z-40">
        {/* ✅ ส่ง `theme` และ `setTheme` ไปที่ Sidebar */}
        <Sidebar activeCourse="Python Series" theme={theme} setTheme={setTheme} sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen} />
      </div>

      {/* ✅ Mobile Sidebar (แสดงเมื่อกด Hamburger Menu) */}
      {mobileMenuOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50">
          {/* ✅ ส่ง `theme` และ `setTheme` ให้ MobileMenu */}
          <MobileMenu onClose={() => setMobileMenuOpen(false)} theme={theme} setTheme={setTheme} />
        </div>
      )}

      {/* ✅ Main Content */}
      <main className="flex-1 md:ml-64 p-4 md:p-6 overflow-y-auto pt-20">
        <div className="max-w-3xl mx-auto">
          {/* ✅ เปลี่ยน "Home" เป็นไอคอนบ้านที่สวยงาม */}
          <div className="flex items-center gap-2 mb-4">
            <button
              className={`p-2 rounded-md flex items-center gap-2 ${
                theme === "dark" ? "bg-gray-700 text-white" : "bg-gray-300 text-black"
              }`}
              onClick={() => navigate("/")}
            >
              <FaHome size={18} className="text-grey" /> {/* ✅ ไอคอนบ้านสีเหลือง */}
            </button>
            <FaAngleRight className="text-gray-400" />
            <span className={`px-3 py-1 rounded-md ${theme === "dark" ? "bg-gray-700" : "bg-gray-300"}`}>
              Python Series
            </span>
          </div>

          <h1 className="text-3xl md:text-4xl font-bold mt-4">Python Series</h1>

          <div className={`p-3 rounded-md mt-4 ${theme === "dark" ? "bg-yellow-600 text-black" : "bg-yellow-400 text-black"}`}>
            ⚠ WARNING: เอกสารนี้อาจมีการเปลี่ยนแปลงตามเนื้อหาของหลักสูตร
          </div>

          {/* ✅ Responsive Table */}
          <div className="overflow-x-auto mt-4">
            <table className={`w-full border rounded-lg shadow-lg ${theme === "dark" ? "border-gray-700" : "border-gray-300"}`}>
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
                  <tr key={lesson.id} className={`border-t ${theme === "dark" ? "border-gray-700" : "border-gray-300"}`}>
                    <td className="p-2">{lesson.id}</td>
                    <td className="p-2">{lesson.title}</td>
                    <td className="p-2">
                      <img src={lesson.image} className="w-20 rounded-lg cursor-pointer" alt={lesson.title} />
                    </td>
                    <td className="p-2">
                      <a href={lesson.docLink} className="text-green-400 hover:underline">อ่าน</a>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* ✅ GitHub Comments Component */}
          <Comments theme={theme} />
        </div>
      </main>

      <SupportMeButton />
    </div>
  );
};

export default PythonSeries;
