import React, { useState, useEffect } from "react";
import { useNavigate, Outlet, useParams } from "react-router-dom"; // ✅ ใช้ Outlet และ useParams
import Navbar from "../../components/common/Navbar";
import PythonSidebar from "../../components/common/sidebar/PythonSidebar";
import SupportMeButton from "../../support/SupportMeButton";
import Comments from "../../components/common/Comments";
import PythonMobileMenu from "../../components/common/sidebar/MobileMenus/PythonMobileMenu";
import Breadcrumb from "../../components/common/Breadcrumb";
import CourseTags from "../../components/common/CourseTags";

// ✅ เปลี่ยนเส้นทางของการ import หลังจากแยกโฟลเดอร์ใหม่
import PythonIntro from "./topics/python/101_basic_python/PythonIntro";
import PythonVariables from "./topics/python/101_basic_python/PythonVariables";
import PythonControlStructure from "./topics/python/101_basic_python/PythonControlStructure";
import PythonInputFunction from "./topics/python/101_basic_python/PythonInputFunction";
import PythonLeetcode from "./topics/python/101_basic_python/PythonLeetcode";

import ListsTuples from "./topics/python/201_data/ListsTuples";
import Dictionaries from "./topics/python/201_data/Dictionaries";
import SetsFrozenset from "./topics/python/201_data/SetFrozenset";
import PandasData from "./topics/python/201_data/PandasData";


import MatplotlibBasics from "./topics/python/202_visualization/MatplotlibBasics";
import SeabornDataVisualization from "./topics/python/202_visualization/SeabornDataVisualization";
import PlotlyInteractiveGraphs from "./topics/python/202_visualization/PlotlyInteractiveGraphs";

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

  

  const topics = [
    { path: "intro", title: "แนะนำ Python" },
    { path: "variables", title: "Variable & Data Type" },
    { path: "control-structure", title: "Control Structure" },
    { path: "input-function", title: "การรับ input & function" },
    { path: "leetcode", title: "LeetCode Challenge" },
    { path: "data", title: "Lists & Tuples" },
    { path: "dictionaries", title: "Dictionaries" },
    { path: "set", title: "Set & Frozenset" },
    { path: "pandas", title: "การจัดการข้อมูลด้วย Pandas" },
    { path: "matplotlib", title: "Matplotlib Basics" },
    { path: "seaborn", title: "Seaborn: Data Visualization" },
    { path: "plotly", title: "Plotly: Interactive Graphs" },
    { path: "data-cleaning", title: "การล้างข้อมูล" },
    { path: "data-transformation", title: "การแปลงข้อมูล" },
    { path: "data-formatting", title: "การจัดรูปแบบข้อมูล" },
    { path: "basic-statistics", title: "สถิติพื้นฐาน" },
    { path: "probability", title: "Probability & Distribution" },
    { path: "hypothesis-testing", title: "Hypothesis Testing" },
    { path: "regression", title: "Regression Analysis" },
    { path: "clustering", title: "Clustering Methods" },
    { path: "deep-learning", title: "Deep Learning Basics" },
  ];

    // ✅ หา index ของหัวข้อปัจจุบัน
    const currentIndex = topics.findIndex((topic) => topic.path === subPage);
    const prevTopic = currentIndex > 0 ? topics[currentIndex - 1] : null;
    const nextTopic = currentIndex < topics.length - 1 ? topics[currentIndex + 1] : null;
  

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

           {/* ✅ เพิ่ม max-height และ scroll bar ถ้าข้อมูลยาวเกิน */}
    <div className="max-h-[75vh] overflow-y-auto  bg-gray-900 rounded-lg">
     {/* ✅ แสดงเนื้อหาหัวข้อย่อย (ProbabilityDistribution.jsx) */}
      </div>

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
{/* ✅ Tags อยู่ในช่วงกรอบสีแดง */}
{/* ✅ ใช้ flex + max-w-5xl mx-auto เพื่อให้ Tags ตรงกับปุ่ม Next */}
<div className="flex justify-between items-center max-w-5xl mx-auto px-4 mt-4">
  <div className="flex items-center">
    <span className="text-lg font-bold">Tags:</span>
    <button
      onClick={() => navigate("/tags/python")}
      className="ml-2 px-3 py-1 border border-gray-500 rounded-lg text-green-700 cursor-pointer hover:bg-gray-700 transition"
    >
      python
    </button>
  </div>
</div>

{/* ✅ ปุ่ม Previous & Next */}
<div className="mt-8 flex justify-between items-center max-w-5xl mx-auto px-4 gap-4">
  {prevTopic ? (
    <button
      className="flex flex-col items-start justify-center w-full max-w-xs md:max-w-sm lg:max-w-md min-w-[150px] min-h-[60px] bg-gray-800 text-white px-6 py-4 rounded-md hover:bg-gray-700 border border-gray-600"
      onClick={() => navigate(`/courses/python-series/${prevTopic.path}`)}
    >
      <span className="text-sm text-gray-400">Previous</span>
      <span className="text-lg">« {prevTopic.title}</span>
    </button>
  ) : (
    <div className="w-full max-w-xs md:max-w-sm lg:max-w-md min-w-[150px] min-h-[60px]"></div>
  )}

  {nextTopic ? (
    <button
      className="flex flex-col items-end justify-center w-full max-w-xs md:max-w-sm lg:max-w-md min-w-[150px] min-h-[60px] bg-gray-800 text-white px-6 py-4 rounded-md hover:bg-gray-700 border border-gray-600"
      onClick={() => navigate(`/courses/python-series/${nextTopic.path}`)}
    >
      <span className="text-sm text-gray-400">Next</span>
      <span className="text-lg">{nextTopic.title} »</span>
    </button>
  ) : (
    <div className="w-full max-w-xs md:max-w-sm lg:max-w-md min-w-[150px] min-h-[60px]"></div>
  )}
</div>

      </main>

      <SupportMeButton />
    </div>
  );
};

export default PythonSeries;
