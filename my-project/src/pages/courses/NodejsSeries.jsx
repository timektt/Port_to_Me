import React, { useState, useEffect } from "react";
import { useNavigate, Outlet, useParams } from "react-router-dom";
import Navbar from "../../components/common/Navbar";
import SupportMeButton from "../../support/SupportMeButton";
import Comments from "../../components/common/Comments";
import Breadcrumb from "../../components/common/Breadcrumb";

// ✅ ตรวจสอบว่าไฟล์ NodeSidebar และ NodeMobileMenu มีอยู่จริง
import NodeSidebar from "../../components/common/sidebar/NodeJsSidebar"; 
import NodeMobileMenu from "../../components/common/sidebar/MobileMenus/NodeMobileMenu";


import NodeIntro from "./topics/nodejs/101_node_basics/NodeIntro.jsx";
import NodeSetup from "./topics/nodejs/101_node_basics/NodeSetup.jsx";
import NodeJsRunCode from "./topics/nodejs/101_node_basics/NodeJsRunCode.jsx";
import NodeModules from "./topics/nodejs/101_node_basics/NodeModules.jsx";
import NodePackageManager from "./topics/nodejs/101_node_basics/NodePackageManager.jsx";

// Node.js 201
import AsyncCallbacks from "./topics/nodejs/201_async_js/AsyncCallbacks.jsx";
import PromisesAsyncAwait from "./topics/nodejs/201_async_js/PromisesAsyncAwait.jsx";
import EventEmitter from "./topics/nodejs/201_async_js/EventEmitter.jsx";
import StreamsBuffer from "./topics/nodejs/201_async_js/StreamsBuffer.jsx";

// Node.js 202
import EventLoop from "./topics/nodejs/202_event_loop/EventLoop.jsx";
import TimersIO from "./topics/nodejs/202_event_loop/TimersIO.jsx";
import ProcessNextTick from "./topics/nodejs/202_event_loop/ProcessNextTick.jsx";

// Node.js 203
import RestApiBasics from "./topics/nodejs/203_api_development/RestApiBasics.jsx";
import HandlingHttpRequests from "./topics/nodejs/203_api_development/HandlingHttpRequests.jsx";
import MiddlewareConcepts from "./topics/nodejs/203_api_development/MiddlewareConcepts.jsx";

// Node.js 204
import ExpressIntro from "./topics/nodejs/204_express/ExpressIntro.jsx";
import ExpressRouting from "./topics/nodejs/204_express/ExpressRouting.jsx";
import ExpressMiddleware from "./topics/nodejs/204_express/ExpressMiddleware.jsx";

// Node.js 205
import MongoDBIntegration from "./topics/nodejs/205_database/MongoDBIntegration.jsx";
import PostgreSQLIntegration from "./topics/nodejs/205_database/PostgreSQLIntegration.jsx";
import MongooseORM from "./topics/nodejs/205_database/MongooseORM.jsx";
import KnexJSPostgreSQL from "./topics/nodejs/205_database/KnexJSPostgreSQL.jsx";

const lessons = [
  { id: "101", title: "Basic Node.js", image: "/node1.jpg", docLink: "/courses/nodejs-series/node-intro", videoLink: "#" },
  { id: "201", title: "Asynchronous JavaScript", image: "/node2.jpg", docLink: "/courses/nodejs-series/async-callbacks", videoLink: "#" },
  { id: "202", title: "Event Loop & Async", image: "/node3.webp", docLink: "/courses/nodejs-series/event-loop", videoLink: "#" },
  { id: "203", title: "API Development", image: "/node1.jpg", docLink: "/courses/nodejs-series/rest-api-basics", videoLink: "#" },
  { id: "204", title: "Express.js", image: "/node2.jpg", docLink: "/courses/nodejs-series/express-intro", videoLink: "#" },
  { id: "205", title: "Database", image: "/node3.webp", docLink: "/courses/nodejs-series/mongodb-integration", videoLink: "#" },
];


const NodeSeries = ({ theme, setTheme }) => {
  const navigate = useNavigate();
  const { "*": subPage } = useParams(); // ✅ เช็คว่าตอนนี้อยู่ในหัวข้อย่อยอะไร
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const topics = [
    { path: "node-intro", title: "Introduction to Node.js" },
    { path: "node-setup", title: "Setting Up Node.js" },
    { path: "node-run-code", title: "Running JavaScript in Node.js" },
    { path: "node-modules", title: "Understanding Modules & require()" },
    { path: "node-npm-yarn", title: "Node Package Manager (NPM & Yarn)" },
    
    { path: "async-callbacks", title: "Understanding Async & Callbacks" },
    { path: "promises-async-await", title: "Promises & Async/Await" },
    { path: "event-emitter", title: "Event Emitter in Node.js" },
    { path: "streams-buffer", title: "Stream & Buffer" },
    { path: "fs-promises", title: "Using fs.promises for File System" },
  
    { path: "event-loop", title: "Understanding the Event Loop" },
    { path: "timers-io", title: "Working with Timers & I/O" },
    { path: "async-error-handling", title: "Handling Asynchronous Errors" },
    { path: "process-next-tick", title: "Using Process & Next Tick" },
    { path: "child-processes", title: "Working with Child Processes" },
  
    { path: "rest-api-basics", title: "Building RESTful APIs" },
    { path: "handling-http-requests", title: "Handling HTTP Requests" },
    { path: "middleware-concepts", title: "Working with Middleware" },
    { path: "error-handling", title: "Data Validation & Error Handling" },
    { path: "api-authentication", title: "Implementing Authentication & JWT" },
  
    { path: "express-intro", title: "Introduction to Express.js" },
    { path: "express-routing", title: "Routing in Express.js" },
    { path: "express-middleware", title: "Handling Middleware in Express" },
    { path: "express-static-files", title: "Serving Static Files" },
    { path: "express-cors", title: "Express.js & CORS" },
  
    { path: "mongodb-integration", title: "Connecting to MongoDB" },
    { path: "postgresql-integration", title: "Using PostgreSQL with Node.js" },
    { path: "mongoose-orm", title: "Working with Mongoose" },
    { path: "knexjs-postgresql", title: "Using Knex.js for SQL Databases" },
    { path: "redis-integration", title: "Using Redis for Caching" },
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
        {NodeSidebar && <NodeSidebar theme={theme} sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen} />}
      </div>


                {/* ✅ Mobile Sidebar */}
                {mobileMenuOpen && (
                  <div className="fixed inset-0 bg-black bg-opacity-50 z-50">
        
                    <NodeMobileMenu onClose={() => setMobileMenuOpen(false)} theme={theme} setTheme={setTheme} />
                  </div>
                )}


      {/* ✅ Main Content */}
      <main className="flex-1 md:ml-64 p-4 md:p-6 mt-16 relative z-10">
        <div className="max-w-5xl mx-auto">
          {/* ✅ Breadcrumb Navigation */}
          <Breadcrumb courseName="Node.js Series" theme={theme} />

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
      onClick={() => navigate("/tags/nodejs")}
      className="ml-2 px-3 py-1 border border-gray-500 rounded-lg text-green-700 cursor-pointer hover:bg-gray-700 transition"
    >
      nodejs
    </button>
  </div>
</div>

{/* ✅ ปุ่ม Previous & Next */}
<div className="mt-8 flex justify-between items-center max-w-5xl mx-auto px-4 gap-4">
  {prevTopic ? (
    <button
      className="flex flex-col items-start justify-center w-full max-w-xs md:max-w-sm lg:max-w-md min-w-[150px] min-h-[60px] bg-gray-800 text-white px-6 py-4 rounded-md hover:bg-gray-700 border border-gray-600"
      onClick={() => navigate(`/courses/nodejs-series/${prevTopic.path}`)}
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
      onClick={() => navigate(`/courses/nodejs-series/${nextTopic.path}`)}
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


export default NodeSeries;