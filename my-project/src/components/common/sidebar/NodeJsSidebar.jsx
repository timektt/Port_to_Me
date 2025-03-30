import React, { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { FaTimes, FaChevronDown, FaChevronRight } from "react-icons/fa";

const sidebarItems = [
  {
    id: "101",
    title: "101: Node.js Basics",
    subItems: [
      { id: "101-1", title: "Introduction to Node.js", path: "/courses/nodejs-series/node-intro" },
      { id: "101-2", title: "Setting Up Node.js", path: "/courses/nodejs-series/node-setup" },
      { id: "101-3", title: "Running JavaScript in Node.js", path: "/courses/nodejs-series/node-run-code" },
      { id: "101-4", title: "Understanding Modules & require()", path: "/courses/nodejs-series/node-modules" },
      { id: "101-5", title: "Node Package Manager (NPM & Yarn)", path: "/courses/nodejs-series/node-npm-yarn" },
    ],
  },
  {
    id: "201",
    title: "201: Asynchronous JavaScript",
    subItems: [
      { id: "201-1", title: "Understanding Async & Callbacks", path: "/courses/nodejs-series/async-callbacks" },
      { id: "201-2", title: "Promises & Async/Await", path: "/courses/nodejs-series/promises-async-await" },
      { id: "201-3", title: "Event Emitter in Node.js", path: "/courses/nodejs-series/event-emitter" },
      { id: "201-4", title: "Stream & Buffer", path: "/courses/nodejs-series/streams-buffer" },
      { id: "201-5", title: "Using fs.promises for File System", path: "/courses/nodejs-series/fs-promises" },
    ],
  },
  {
    id: "202",
    title: "202: Event Loop & Async Operations",
    subItems: [
      { id: "202-1", title: "Understanding the Event Loop", path: "/courses/nodejs-series/event-loop" },
      { id: "202-2", title: "Working with Timers & I/O", path: "/courses/nodejs-series/timers-io" },
      { id: "202-3", title: "Handling Asynchronous Errors", path: "/courses/nodejs-series/async-error-handling" },
      { id: "202-4", title: "Using Process & Next Tick", path: "/courses/nodejs-series/process-next-tick" },
      { id: "202-5", title: "Working with Child Processes", path: "/courses/nodejs-series/child-processes" },
    ],
  },
  {
    id: "203",
    title: "203: API Development",
    subItems: [
      { id: "203-1", title: "Building RESTful APIs", path: "/courses/nodejs-series/rest-api-basics" },
      { id: "203-2", title: "Handling HTTP Requests", path: "/courses/nodejs-series/handling-http-requests" },
      { id: "203-3", title: "Working with Middleware", path: "/courses/nodejs-series/middleware-concepts" },
      { id: "203-4", title: "Data Validation & Error Handling", path: "/courses/nodejs-series/error-handling" },
      { id: "203-5", title: "Implementing Authentication & JWT", path: "/courses/nodejs-series/api-authentication" },
    ],
  },
  {
    id: "204",
    title: "204: Express.js Framework",
    subItems: [
      { id: "204-1", title: "Introduction to Express.js", path: "/courses/nodejs-series/express-intro" },
      { id: "204-2", title: "Routing in Express.js", path: "/courses/nodejs-series/express-routing" },
      { id: "204-3", title: "Handling Middleware in Express", path: "/courses/nodejs-series/express-middleware" },
      { id: "204-4", title: "Serving Static Files", path: "/courses/nodejs-series/express-error-handling" },
      { id: "204-5", title: "Express.js & CORS", path: "/courses/nodejs-series/express-cors" },
    ],
  },
  {
    id: "205",
    title: "205: Database Integration",
    subItems: [
      { id: "205-1", title: "Connecting to MongoDB", path: "/courses/nodejs-series/mongodb-integration" },
      { id: "205-2", title: "Using PostgreSQL with Node.js", path: "/courses/nodejs-series/postgresql-integration" },
      { id: "205-3", title: "Working with Mongoose", path: "/courses/nodejs-series/mongoose-orm" },
      { id: "205-4", title: "Using Knex.js for SQL Databases", path: "/courses/nodejs-series/knexjs-postgresql" },
      { id: "205-5", title: "Using Redis for Caching", path: "/courses/nodejs-series/redis-integration" },
    ],
  },
];


const NodeJsSidebar = ({ theme, sidebarOpen, setSidebarOpen }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const [expandedSections, setExpandedSections] = useState({});

  const toggleSection = (id) => {
    setExpandedSections((prev) => ({
      ...prev,
      [id]: !prev[id],
    }));
  };

  return (
    <>
      {sidebarOpen && (
        <div className="fixed inset-0 bg-black/40 backdrop-blur-sm z-40 md:hidden" onClick={() => setSidebarOpen(false)} />
      )}
  
      <aside
        className={`fixed top-16 left-0 w-64 h-[calc(100vh-70px)] overflow-y-auto z-50 p-4 transition-transform duration-300 ease-in-out
          ${sidebarOpen ? "translate-x-0" : "-translate-x-full"} md:translate-x-0
          ${theme === "dark" ? "bg-gray-900 text-white" : "bg-white text-black"} shadow-lg pb-20`} // ✅ เพิ่ม pb-20
      >
        <button className="md:hidden absolute top-4 right-4 text-xl" onClick={() => setSidebarOpen(false)}>
          <FaTimes />
        </button>
  
         
        {/* Header: NodeJs Series Title (Clickable) */}
        <h2
          className="text-xl font-bold mb-4 cursor-pointer transition hover:underline hover:text-yellow-400"
          onClick={() => navigate("/courses/nodejs-series")}
        >
          <span
            className={`inline-block px-3 py-1 rounded-md ${
              theme === "dark" ? "bg-gray-700" : "bg-gray-200"
            }`}
          >
            NodeJs Series
          </span>
        </h2>
  
        <ul className="space-y-2 mt-4 mb-24">  {/* ✅ เพิ่ม mb-24 เพื่อให้ Scroll ได้ */}
          {sidebarItems.map((item) => (
            <li key={item.id} className="border-b border-gray-700">
              <button
                className="flex items-center justify-between w-full p-3 rounded-lg transition duration-300 ease-in-out
                  hover:bg-gray-700 hover:shadow-lg text-left"
                onClick={() => toggleSection(item.id)}
              >
                {item.title}
                {expandedSections[item.id] ? <FaChevronDown /> : <FaChevronRight />}
              </button>
  
              {expandedSections[item.id] && (
                <ul className="pl-5 space-y-2 mt-2">
                  {item.subItems.map((subItem) => (
                    <li
                      key={subItem.id}
                      className={`p-2 rounded-lg cursor-pointer transition duration-200 ${
                        location.pathname === subItem.path ? "bg-green-500 text-white font-bold" : "hover:bg-gray-600"
                      }`}
                      onClick={() => navigate(subItem.path)}
                    >
                      {subItem.title}
                    </li>
                  ))}
                </ul>
              )}
            </li>
          ))}
        </ul>
      </aside>
    </>
  );
  
};

export default NodeJsSidebar;
