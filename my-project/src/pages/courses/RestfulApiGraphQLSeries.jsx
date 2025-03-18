import React, { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import { useNavigate } from "react-router-dom";
import Navbar from "../../components/common/Navbar";
import RestfulApiGraphQLSidebar from "../../components/common/sidebar/RestfulApiGraphQLSidebar"; // ✅ Sidebar ใหม่
import SupportMeButton from "../../support/SupportMeButton";
import Comments from "../../components/common/Comments";
import RestfulApiGraphQLMobileMenu from "../../components/common/sidebar/MobileMenus/RestfulApiGraphQLMobileMenu"; // ✅ Mobile Menu ใหม่
import Breadcrumb from "../../components/common/Breadcrumb"; // ✅ ใช้ Breadcrumb

const lessons = [
  // 101: Introduction to API
  { id: "101", title: "What is an API?", image: "/api1.webp", docLink: "#", videoLink: "#" },
 
  // 201: RESTful API
  { id: "201", title: "RESTful API Basics", image: "/api2.jpg", docLink: "#", videoLink: "#" },
 
  // 202: GraphQL
  { id: "202", title: "GraphQL Basics", image: "/api3.png", docLink: "#", videoLink: "#" },

  // 203: API Security
  { id: "203", title: "Authentication & Authorization", image: "/api1.webp", docLink: "#", videoLink: "#" },

  // 204: Advanced API Concepts
  { id: "204", title: "API Gateways & Microservices", image: "/api2.jpg", docLink: "#", videoLink: "#" },
 
];


const RestfulApiGraphQLSeries = ({ theme, setTheme }) => {
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

      {/* ✅ Sidebar (เลื่อนลงให้ไม่ทับ Navbar) */}
      <div className="hidden md:block fixed left-0 top-16 h-[calc(100vh-4rem)] w-64 z-40">
        {RestfulApiGraphQLSidebar && <RestfulApiGraphQLSidebar theme={theme} sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen} />}
      </div>


                {/* ✅ Mobile Sidebar */}
                {mobileMenuOpen && (
                  <div className="fixed inset-0 bg-black bg-opacity-50 z-50">
                    {console.log("✅ NodeMobileMenu Rendered!")}
                    <NodeMobileMenu onClose={() => setMobileMenuOpen(false)} theme={theme} setTheme={setTheme} />
                  </div>
                )}


      {/* ✅ Main Content */}ฆ
      <main className="flex-1 md:ml-64 p-4 md:p-6 mt-16 relative z-10">
        <div className="max-w-5xl mx-auto">
          {/* ✅ Breadcrumb Navigation */}
          <Breadcrumb courseName="RESTful API & GraphQL" theme={theme} />

          {/* ✅ ถ้ามี subPage แสดงเนื้อหาเฉพาะหน้านั้น */}
          {subPage ? (
            <Outlet /> // ✅ โหลดเนื้อหาหัวข้อย่อยที่เลือก
          ) : (
            <>
              <h1 className="text-3xl md:text-4xl font-bold mt-4">Node.js Series</h1>

              {/* ✅ Warning Box */}
              <div className={`p-4 mt-4 rounded-md shadow-md flex flex-col gap-2 ${theme === "dark" ? "bg-yellow-700 text-white" : "bg-yellow-300 text-black"}`}>
                <strong className="text-lg flex items-center gap-2">⚠ WARNING</strong>
                <p>เอกสารฉบับนี้ยังอยู่ในระหว่างการทำ Series ของ Node.js...</p>
                <p>สามารถติดตามผ่านทาง Youtube: <a href="https://youtube.com" className="text-blue-400 hover:underline ml-1">supermhee</a></p>
              </div>

              {/* ✅ Table Section (Desktop) */}
              <div className="hidden sm:block overflow-x-auto mt-6">
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
                        <button 
                          onClick={() => navigate(lesson.docLink)}
                          className="text-green-400 hover:underline hover:text-green-500"
                        >
                          อ่าน
                        </button>
                      </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* ✅ Responsive (Mobile) */}
              <div className="sm:hidden mt-6 space-y-4">
                {lessons.map((lesson) => (
                  <div key={lesson.id} className={`p-4 border rounded-lg shadow-md ${theme === "dark" ? "bg-gray-800 text-white" : "bg-white text-black"}`}>
                    <h2 className="text-xl font-semibold">{lesson.title}</h2>
                    <img src={lesson.image} className="w-full h-40 mt-2 rounded-lg shadow-md object-cover" alt={lesson.title} />
                    <div className="mt-4 flex justify-between">
                      <a href={lesson.videoLink} target="_blank" rel="noopener noreferrer" className="text-green-400 hover:underline">ดู video</a>
                      <a href={lesson.docLink} target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">อ่านเอกสาร</a>
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
export default RestfulApiGraphQLSeries;