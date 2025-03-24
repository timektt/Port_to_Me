import React, { useState, useEffect } from "react";
import { Outlet } from "react-router-dom";
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
  { id: "101", title: "What is an API?", image: "/api1.webp", docLink: "/courses/restful-api-graphql-series/intro", videoLink: "#" },
 
  // 201: RESTful API
  { id: "201", title: "RESTful API Basics", image: "/api2.jpg", docLink: "/courses/restful-api-graphql-series/rest-basics", videoLink: "#" },
 
  // 202: GraphQL
  { id: "202", title: "GraphQL Basics", image: "/api3.png", docLink: "/courses/restful-api-graphql-series/graphql-basics", videoLink: "#" },

  // 203: API Security
  { id: "203", title: "Authentication & Authorization", image: "/api1.webp", docLink: "/courses/restful-api-graphql-series/api-security", videoLink: "#" },

  // 204: Advanced API Concepts
  { id: "204", title: "API Gateways & Microservices", image: "/api2.jpg", docLink: "/courses/restful-api-graphql-series/api-gateways-microservices", videoLink: "#" },
 
];


const RestfulApiGraphQLSeries = ({ theme, setTheme }) => {
  const navigate = useNavigate();
  const { "*": subPage } = useParams(); // ✅ เช็คว่าตอนนี้อยู่ในหัวข้อย่อยอะไร
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const topics = [
    // ✅ 101: Introduction to API
    { path: "intro", title: "What is an API?" },
    { path: "rest-vs-graphql", title: "REST vs GraphQL" },
    { path: "how-apis-work", title: "How APIs Work" },
    { path: "api-types", title: "Types of APIs (Public, Private, Partner, Composite)" },
    { path: "api-documentation", title: "API Documentation & Tools" },

    // ✅ 201: RESTful API
    { path: "rest-basics", title: "RESTful API Basics" },
    { path: "rest-nodejs", title: "Building RESTful API with Node.js" },
    { path: "rest-crud", title: "CRUD Operations in REST" },
    { path: "rest-error-handling", title: "Handling Errors in REST API" },
    { path: "rest-versioning", title: "Versioning in REST API" },

    // ✅ 202: GraphQL
    { path: "graphql-basics", title: "GraphQL Basics" },
    { path: "graphql-api", title: "Building GraphQL API" },
    { path: "graphql-queries-mutations", title: "Queries and Mutations" },
    { path: "graphql-schema-resolvers", title: "GraphQL Schema & Resolvers" },
    { path: "graphql-vs-rest", title: "GraphQL vs REST: Pros & Cons" },

    // ✅ 203: API Security
    { path: "api-security", title: "Authentication & Authorization" },
    { path: "rate-limiting", title: "Rate Limiting & CORS" },
    { path: "oauth-api-keys", title: "OAuth & API Keys" },
    { path: "jwt-session", title: "JWT & Session Management" },
    { path: "api-security-best-practices", title: "Best Practices for API Security" },

    // ✅ 204: Advanced API Concepts
    { path: "api-gateways-microservices", title: "API Gateways & Microservices" },
    { path: "graphql-subscriptions", title: "GraphQL Subscriptions (Real-time API)" },
    { path: "api-performance", title: "API Performance Optimization" },
    { path: "api-testing-monitoring", title: "Testing & Monitoring APIs" },
    { path: "api-deployment-scaling", title: "Deploying & Scaling APIs" },
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
              <h1 className="text-3xl md:text-4xl font-bold mt-4">RestfulApiGraphQLSeries</h1>

              {/* ✅ Warning Box */}
              <div className={`p-4 mt-4 rounded-md shadow-md flex flex-col gap-2 ${theme === "dark" ? "bg-yellow-700 text-white" : "bg-yellow-300 text-black"}`}>
                <strong className="text-lg flex items-center gap-2">⚠ WARNING</strong>
                <p>เอกสารฉบับนี้ยังอยู่ในระหว่างการทำ Series ของ Node.js...</p>
                <p>สามารถติดตามผ่านทาง Youtube: <a href="https://youtube.com" className="text-blue-400 hover:underline ml-1">Superbear</a></p>
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
            {/* ✅ Tags อยู่ในช่วงกรอบสีแดง */}
{/* ✅ ใช้ flex + max-w-5xl mx-auto เพื่อให้ Tags ตรงกับปุ่ม Next */}
<div className="flex justify-between items-center max-w-5xl mx-auto px-4 mt-4">
  <div className="flex items-center">
    <span className="text-lg font-bold">Tags:</span>
    <button
      onClick={() => navigate("/tags/graphqL")}
      className="ml-2 px-3 py-1 border border-gray-500 rounded-lg text-green-700 cursor-pointer hover:bg-gray-700 transition"
    >
      restfulapi&graphql
    </button>
  </div>
</div>

{/* ✅ ปุ่ม Previous & Next */}
<div className="mt-8 flex justify-between items-center max-w-5xl mx-auto px-4 gap-4">
  {prevTopic ? (
    <button
      className="flex flex-col items-start justify-center w-full max-w-xs md:max-w-sm lg:max-w-md min-w-[150px] min-h-[60px] bg-gray-800 text-white px-6 py-4 rounded-md hover:bg-gray-700 border border-gray-600"
      onClick={() => navigate(`/courses/restful-api-graphql-series/${prevTopic.path}`)}
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
      onClick={() => navigate(`/courses/restful-api-graphql-series/${nextTopic.path}`)}
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
export default RestfulApiGraphQLSeries;