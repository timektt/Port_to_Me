import React, { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import {
  FaTimes,
  FaChevronDown,
  FaChevronRight,
  FaArrowLeft,
  FaMoon,
} from "react-icons/fa";
import { FiSun } from "react-icons/fi"; // ✅ ใช้ FiSun แทน FaSun

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

const NodeMobileMenu = ({ onClose, theme, setTheme }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const [expandedSections, setExpandedSections] = useState({});

  const toggleTheme = () => {
    const newTheme = theme === "dark" ? "light" : "dark";
    setTheme(newTheme);
    localStorage.setItem("theme", newTheme);
  };

  const toggleSection = (id) => {
    setExpandedSections((prev) => ({
      ...prev,
      [id]: !prev[id],
    }));
  };

  return (
    <div
      className={`fixed top-0 left-0 w-64 h-full p-4 z-50 shadow-lg transition-all duration-300 ${
        theme === "dark" ? "bg-gray-900 text-white" : "bg-white text-black"
      } overflow-y-auto pb-20`}
    >
      {/* ✅ ปุ่มปิดเมนู (X) */}
      <button
        className={`absolute right-4 top-4 text-2xl transition-colors duration-200 ${
          theme === "dark"
            ? "text-white hover:text-gray-400"
            : "text-black hover:text-gray-600"
        }`}
        onClick={onClose}
      >
        <FaTimes />
      </button>

      {/* ✅ โลโก้ + Superbear + ปุ่ม Dark/Light Mode */}
      <div className="mt-6 flex items-center mb-3">
        <img
          src="/spm2.jpg"
          alt="Logo"
          className="w-8 h-8 mr-2 object-cover rounded-full"
        />
        <div className="flex items-center space-x-2">
          <span className="text-lg font-bold cursor-pointer hover:text-gray-400 transition">
            Superbear
          </span>
          <button
            className="cursor-pointer transition-transform transform hover:scale-110"
            onClick={toggleTheme}
          >
            {theme === "dark" ? (
              <FiSun className="text-yellow-400 text-2xl" /> // ✅ ใช้ FiSun
            ) : (
              <FaMoon className="text-blue-400 text-2xl" />
            )}
          </button>
        </div>
      </div>

      {/* ✅ ปุ่มกลับไปหน้า Node.js Series */}
      <button
        className={`w-full text-left text-sm font-medium px-5 py-3 rounded-lg mb-4 transition 
          ${theme === "dark" ? "bg-gray-800 text-white hover:bg-gray-700" : "bg-gray-200 text-black hover:bg-gray-300"}`}
        onClick={() => {
          navigate("/courses/nodejs-series");
          onClose();
        }}
      >
        <FaArrowLeft className="inline-block mr-2" /> Node.js Series
      </button>

      {/* ✅ รายการบทเรียน (Dropdown) */}
      <ul className="space-y-2 mt-4">
        {sidebarItems.map((item) => (
          <li key={item.id} className="border-b border-gray-700">
            <button
              className="flex items-center justify-between w-full p-4 rounded-lg transition duration-300 ease-in-out
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
                      location.pathname === subItem.path
                        ? "bg-green-500 text-white font-bold"
                        : "hover:bg-gray-600"
                    }`}
                    onClick={() => {
                      navigate(subItem.path);
                      onClose();
                    }}
                  >
                    {subItem.title}
                  </li>
                ))}
              </ul>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default NodeMobileMenu;
