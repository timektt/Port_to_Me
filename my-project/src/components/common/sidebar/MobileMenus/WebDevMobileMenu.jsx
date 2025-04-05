import React, { useState } from "react";
import { FaTimes, FaChevronDown, FaChevronRight, FaArrowLeft, FaMoon } from "react-icons/fa";
import { FiSun } from "react-icons/fi"; // ✅ ใช้ FiSun แทน FaSun
import { useNavigate, useLocation } from "react-router-dom";

const sidebarItems = [
  {
    id: "101",
    title: "101: Web Development Basics",
    subItems: [
      { id: "101-1", title: "Introduction to Web Development", path: "/courses/web-development/intro" },
      { id: "101-2", title: "Frontend vs Backend", path: "/courses/web-development/frontend-backend" },
      { id: "101-3", title: "How the Web Works", path: "/courses/web-development/how-web-works" },
      { id: "101-4", title: "Client vs Server", path: "/courses/web-development/client-server" },
      { id: "101-5", title: "Essential Web Development Tools", path: "/courses/web-development/web-dev-tools" },
    ],
  },
  {
    id: "201",
    title: "201: HTML & CSS",
    subItems: [
      { id: "201-1", title: "HTML Basics", path: "/courses/web-development/html-basics" },
      { id: "201-2", title: "CSS Basics", path: "/courses/web-development/css-basics" },
      { id: "201-3", title: "Responsive Design", path: "/courses/web-development/responsive-design" },
      { id: "201-4", title: "CSS Grid & Flexbox", path: "/courses/web-development/css-grid-flexbox" },
      { id: "201-5", title: "CSS Preprocessors (SASS & LESS)", path: "/courses/web-development/css-preprocessors" },
    ],
  },
  {
    id: "202",
    title: "202: JavaScript for Web",
    subItems: [
      { id: "202-1", title: "JavaScript Basics", path: "/courses/web-development/javascript-basics" },
      { id: "202-2", title: "DOM Manipulation", path: "/courses/web-development/dom-manipulation" },
      { id: "202-3", title: "ES6+ Modern JavaScript", path: "/courses/web-development/es6-modern-js" },
      { id: "202-4", title: "Event Handling", path: "/courses/web-development/event-handling" },
      { id: "202-5", title: "Asynchronous JavaScript (Promises & Async/Await)", path: "/courses/web-development/async-js" },
    ],
  },
  {
    id: "203",
    title: "203: Frontend Frameworks",
    subItems: [
      { id: "203-1", title: "Introduction to React", path: "/courses/web-development/react-intro" },
      { id: "203-2", title: "Vue.js Basics", path: "/courses/web-development/vue-intro" },
      { id: "203-3", title: "Angular Basics", path: "/courses/web-development/angular-intro" },
      { id: "203-4", title: "State Management (Redux, Vuex, Pinia, NgRx)", path: "/courses/web-development/state-management" },
      { id: "203-5", title: "SSR vs CSR", path: "/courses/web-development/ssr-vs-csr" },
    ],
  },
  {
    id: "204",
    title: "204: Backend Development",
    subItems: [
      { id: "204-1", title: "Node.js & Express", path: "/courses/web-development/node-express" },
      { id: "204-2", title: "API Development", path: "/courses/web-development/api-development" },
      { id: "204-3", title: "Authentication & Authorization", path: "/courses/web-development/authentication" },
      { id: "204-4", title: "File Upload & Image Processing", path: "/courses/web-development/file-upload" },
      { id: "204-5", title: "WebSockets & Real-time Applications", path: "/courses/web-development/websockets" },
    ],
  },
  {
    id: "205",
    title: "205: Databases & APIs",
    subItems: [
      { id: "205-1", title: "MongoDB Basics", path: "/courses/web-development/mongodb" },
      { id: "205-2", title: "SQL Fundamentals", path: "/courses/web-development/sql-basics" },
      { id: "205-3", title: "REST & GraphQL APIs", path: "/courses/web-development/rest-graphql" },
      { id: "205-4", title: "Caching Strategies (Redis, Memcached)", path: "/courses/web-development/caching-strategies" },
      { id: "205-5", title: "Database Optimization & Indexing", path: "/courses/web-development/db-optimization" },
    ],
  },
];

const WebDevMobileMenu = ({ onClose, theme, setTheme }) => {
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
          theme === "dark" ? "text-white hover:text-gray-400" : "text-black hover:text-gray-600"
        }`}
        onClick={onClose}
      >
        <FaTimes />
      </button>

      {/* ✅ โลโก้ + Superbear + ปุ่ม Dark/Light Mode */}
      <div className="mt-6 flex items-center mb-3">
        <img src="/spm2.jpg" alt="Logo" className="w-8 h-8 mr-2 object-cover rounded-full" />
        <div className="flex items-center space-x-2">
          <span className="text-lg font-bold cursor-pointer hover:text-gray-400 transition">
            Superbear
          </span>
          <button className="cursor-pointer transition-transform transform hover:scale-110" onClick={toggleTheme}>
            {theme === "dark" ? (
              <FiSun className="text-yellow-400 text-2xl" /> // ✅ ใช้ FiSun
            ) : (
              <FaMoon className="text-blue-400 text-2xl" />
            )}
          </button>
        </div>
      </div>

      {/* ✅ ปุ่มกลับไปหน้า Web Development Series */}
      <button
        className={`w-full text-left text-sm font-medium px-5 py-3 rounded-lg mb-4 transition 
          ${theme === "dark" ? "bg-gray-800 text-white hover:bg-gray-700" : "bg-gray-200 text-black hover:bg-gray-300"}`}
        onClick={() => {
          navigate("/courses/web-development");
          onClose();
        }}
      >
        <FaArrowLeft className="inline-block mr-2" /> Web Development Series
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
                      location.pathname === subItem.path ? "bg-green-500 text-white font-bold" : "hover:bg-gray-600"
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

export default WebDevMobileMenu;
