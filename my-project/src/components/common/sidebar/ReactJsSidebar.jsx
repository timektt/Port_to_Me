import React, { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { FaTimes, FaChevronDown, FaChevronRight } from "react-icons/fa";

const sidebarItems = [
  {
    id: "101",
    title: "101: Introduction to React.js",
    subItems: [
      { id: "101-1", title: "What is React.js?", path: "/courses/reactjs-series/intro" },
      { id: "101-2", title: "Setting Up React Project", path: "/courses/reactjs-series/setup" },
      { id: "101-3", title: "JSX & Rendering", path: "/courses/reactjs-series/jsx-rendering" },
    ],
  },
  {
    id: "201",
    title: "201: React Components & Props",
    subItems: [
      { id: "201-1", title: "Functional & Class Components", path: "/courses/reactjs-series/components" },
      { id: "201-2", title: "Props & Prop Drilling", path: "/courses/reactjs-series/props" },
    ],
  },
  {
    id: "202",
    title: "202: State Management",
    subItems: [
      { id: "202-1", title: "Using useState Hook", path: "/courses/reactjs-series/state" },
      { id: "202-2", title: "React Context API", path: "/courses/reactjs-series/context-api" },
      { id: "202-3", title: "Redux Basics", path: "/courses/reactjs-series/redux" },
    ],
  },
];

const ReactJsSidebar = ({ theme, sidebarOpen, setSidebarOpen }) => {
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
        className={`fixed top-16 left-0 w-64 h-[calc(100vh-64px)] overflow-y-auto z-50 p-4 transition-transform duration-300 ease-in-out
          ${sidebarOpen ? "translate-x-0" : "-translate-x-full"} md:translate-x-0
          ${theme === "dark" ? "bg-gray-900 text-white" : "bg-white text-black"} shadow-lg`}
      >
        <button className="md:hidden absolute top-4 right-4 text-xl" onClick={() => setSidebarOpen(false)}>
          <FaTimes />
        </button>

        <h2 className="text-xl font-bold mb-4">
          <span className={`inline-block px-3 py-1 rounded-md ${theme === "dark" ? "bg-gray-700" : "bg-gray-200"}`}>
            React.js Series
          </span>
        </h2>

        <ul className="space-y-2 mt-4">
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

export default ReactJsSidebar;
