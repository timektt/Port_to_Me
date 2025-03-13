import React, { useState, useEffect } from "react";
import { useNavigate, Outlet, useParams } from "react-router-dom"; // ✅ ใช้ Outlet และ useParams
import Navbar from "../../components/common/Navbar";
import PythonSidebar from "../../components/common/sidebar/PythonSidebar";
import SupportMeButton from "../../support/SupportMeButton";
import Comments from "../../components/common/Comments";
import PythonMobileMenu from "../../components/common/sidebar/MobileMenus/PythonMobileMenu";
import Breadcrumb from "../../components/common/Breadcrumb";

// ✅ เปลี่ยนเส้นทางของการ import หลังจากแยกโฟลเดอร์ใหม่
import PythonIntro from "./topics/python/101_basic_python/PythonIntro";
import PythonVariables from "./topics/python/101_basic_python/PythonVariables";
import PythonControlStructure from "./topics/python/101_basic_python/PythonControlStructure";
import PythonInputFunction from "./topics/python/101_basic_python/PythonInputFunction";
import PythonLeetcode from "./topics/python/101_basic_python/PythonLeetcode";

const lessons = [
  { id: "101", title: "Python Introduction", image: "/py1.png", docLink: "#", videoLink: "#" },
  { id: "201", title: "Basic Data", image: "/py2.jpg", docLink: "#", videoLink: "#" },
  { id: "202", title: "Visualization", image: "/py3.jpg", docLink: "#", videoLink: "#" },
];

const PythonSeries = ({ theme, setTheme }) => {
  const navigate = useNavigate();
  const { "*": subPage } = useParams(); // ✅ เช็คว่าตอนนี้อยู่ในหัวข้อย่อยอะไร
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

      {/* ✅ Sidebar (จะไม่ถูกรีโหลดใหม่) */}
      <div className="hidden md:block fixed left-0 top-16 h-[calc(100vh-4rem)] w-64 z-40">
        <PythonSidebar theme={theme} sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen} />
      </div>

      {/* ✅ Mobile Sidebar */}
      {mobileMenuOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50">
          <PythonMobileMenu onClose={() => setMobileMenuOpen(false)} theme={theme} setTheme={setTheme} />
        </div>
      )}

      {/* ✅ Main Content */}
      <main className="flex-1 md:ml-64 p-4 md:p-6 mt-16 relative z-10">
        <div className="max-w-5xl mx-auto">
          {/* ✅ Breadcrumb Navigation */}
          <Breadcrumb courseName="Python Series" theme={theme} />

          {/* ✅ เช็คว่ามีหัวข้อย่อยใน URL หรือไม่ */}
          {subPage ? (
            <Outlet /> // ✅ แสดงเนื้อหาหัวข้อย่อยที่เลือก
          ) : (
            <>
              <h1 className="text-3xl md:text-4xl font-bold mt-4">Python Series</h1>

              {/* ✅ Warning Box */}
              <div className={`p-4 mt-4 rounded-md shadow-md flex flex-col gap-2 ${theme === "dark" ? "bg-yellow-700 text-white" : "bg-yellow-300 text-black"}`}>
                <strong className="text-lg flex items-center gap-2">⚠ WARNING</strong>
                <p>เอกสารฉบับนี้ยังอยู่ในระหว่างการทำ Series ของ Python...</p>
                <p>
                  สามารถติดตามผ่านทาง Youtube:{" "}
                  <a href="https://youtube.com" className="text-blue-400 hover:underline ml-1">
                    supermhee
                  </a>
                </p>
              </div>

              {/* ✅ Table Section สำหรับหน้าจอใหญ่ */}
              <div className="hidden sm:block overflow-x-auto mt-6">
                <table className={`min-w-full table-auto border rounded-lg shadow-lg ${theme === "dark" ? "border-gray-700" : "border-gray-300"}`}>
                  <thead className={`${theme === "dark" ? "bg-gray-800 text-white" : "bg-gray-300 text-black"} text-lg`}>
                    <tr>
                      <th className="p-4 border-b-2">ตอน</th>
                      <th className="p-4 border-b-2">หัวข้อ</th>
                      <th className="p-4 border-b-2">วิดีโอ</th>
                      <th className="p-4 border-b-2">เอกสาร</th>
                    </tr>
                  </thead>
                  <tbody>
                    {lessons.map((lesson, index) => (
                      <tr key={lesson.id} className={`${index % 2 === 0 ? (theme === "dark" ? "bg-gray-700" : "bg-gray-100") : ""} hover:bg-gray-500 transition duration-200`}>
                        <td className="p-4 text-center border-b text-lg font-semibold">{lesson.id}</td>
                        <td className="p-4 border-b text-lg">{lesson.title}</td>
                        <td className="p-4 border-b text-center">
                          <a href={lesson.videoLink} target="_blank" rel="noopener noreferrer">
                            <img src={lesson.image} className="w-48 h-36 mx-auto rounded-lg shadow-lg cursor-pointer transition-transform transform hover:scale-105 hover:shadow-xl object-cover" alt={lesson.title} />
                            <span className="block mt-2 text-green-400 hover:underline">ดู video</span>
                          </a>
                        </td>
                        <td className="p-4 border-b text-center">
                          <a href={lesson.docLink} target="_blank" rel="noopener noreferrer" className="text-green-400 hover:underline">
                            อ่าน
                          </a>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* ✅ Responsive Section (Mobile) */}
              <div className="sm:hidden mt-6 space-y-4">
                {lessons.map((lesson) => (
                  <div key={lesson.id} className={`p-4 border rounded-lg shadow-md ${theme === "dark" ? "bg-gray-800 text-white" : "bg-white text-black"}`}>
                    <h2 className="text-xl font-semibold">{lesson.title}</h2>
                    <img src={lesson.image} className="w-full h-40 mt-2 rounded-lg shadow-md object-cover" alt={lesson.title} />
                    <div className="mt-4 flex justify-between">
                      <a href={lesson.videoLink} target="_blank" rel="noopener noreferrer" className="text-green-400 hover:underline">
                        ดู video
                      </a>
                      <a href={lesson.docLink} target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">
                        อ่านเอกสาร
                      </a>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}

          {/* ✅ Comments Section */}
          <Comments theme={theme} />
        </div>
      </main>

      <SupportMeButton />
    </div>
  );
};

export default PythonSeries;
