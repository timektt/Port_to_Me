import React, { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { FaTimes, FaChevronDown, FaChevronRight, FaArrowLeft, FaSun, FaMoon } from "react-icons/fa";

const sidebarItems = [
  {
    id: "101",
    title: "101: Basic Node.js",
    subItems: [
      { id: "101-1", title: "Introduction to Node.js", path: "/courses/nodejs-series/node-intro" },
      { id: "101-2", title: "Setting Up Node.js", path: "/courses/nodejs-series/node-setup" },
      { id: "101-3", title: "Running JavaScript in Node.js", path: "/courses/nodejs-series/node-run-code" },
      { id: "101-4", title: "Understanding Modules & require()", path: "/courses/nodejs-series/node-modules" },
      { id: "101-5", title: "Node Package Manager (NPM & Yarn)", path: "/courses/nodejs-series/node-package-manager" },
    ],
  },
  {
    id: "201",
    title: "201: Asynchronous JavaScript",
    subItems: [
      { id: "201-1", title: "Understanding Async & Callbacks", path: "/courses/nodejs-series/async-callbacks" },
      { id: "201-2", title: "Promises & Async/Await", path: "/courses/nodejs-series/promises-async-await" },
      { id: "201-3", title: "Event Emitters & Listeners", path: "/courses/nodejs-series/event-emitter" },
      { id: "201-4", title: "Handling Streams & Buffers", path: "/courses/nodejs-series/streams-buffer" },
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
    <div className={`fixed top-0 left-0 w-64 h-full p-4 z-50 shadow-lg transition-all duration-300 ${theme === "dark" ? "bg-gray-900 text-white" : "bg-white text-black"}`}>
      <button className={`absolute right-4 top-4 text-2xl transition-colors duration-200 ${theme === "dark" ? "text-white hover:text-gray-400" : "text-black hover:text-gray-600"}`} onClick={onClose}>
        <FaTimes />
      </button>

      <div className="mt-6 flex items-center mb-3">
        <span className="text-lg font-bold cursor-pointer hover:text-gray-400 transition">Supermhee</span>
        <button className="cursor-pointer transition-transform transform hover:scale-110" onClick={toggleTheme}>
          {theme === "dark" ? <FaSun className="text-yellow-400 text-xl" /> : <FaMoon className="text-blue-400 text-xl" />}
        </button>
      </div>

      <button className="flex items-center text-sm text-gray-400 hover:text-gray-300 mb-3 transition" onClick={onClose}>
        <FaArrowLeft className="mr-2" /> Back to main menu
      </button>

      <ul className="space-y-2 mt-4">
        {sidebarItems.map((item) => (
          <li key={item.id} className="border-b border-gray-700">
            <button className="flex items-center justify-between w-full p-3 rounded-lg transition duration-300 ease-in-out hover:bg-gray-700 hover:shadow-lg text-left" onClick={() => toggleSection(item.id)}>
              {item.title}
              {expandedSections[item.id] ? <FaChevronDown /> : <FaChevronRight />}
            </button>
            {expandedSections[item.id] && (
              <ul className="pl-5 space-y-2 mt-2">
                {item.subItems.map((subItem) => (
                  <li
                    key={subItem.id}
                    className={`p-2 rounded-lg cursor-pointer transition duration-200 ${location.pathname === subItem.path ? "bg-green-500 text-white font-bold" : "hover:bg-gray-600"}`}
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
