import React, { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { FaTimes, FaChevronDown, FaChevronRight } from "react-icons/fa";

const sidebarItems = [
  {
    id: "101",
    title: "101: Web Development Basics",
    subItems: [
      { id: "101-1", title: "Introduction to Web Development", path: "/courses/web-development/intro" },
      { id: "101-2", title: "Frontend vs Backend", path: "/courses/web-development/frontend-backend" },
    ],
  },
  {
    id: "201",
    title: "201: HTML & CSS",
    subItems: [
      { id: "201-1", title: "HTML Basics", path: "/courses/web-development/html-basics" },
      { id: "201-2", title: "CSS Basics", path: "/courses/web-development/css-basics" },
      { id: "201-3", title: "Responsive Design", path: "/courses/web-development/responsive-design" },
    ],
  },
  {
    id: "202",
    title: "202: JavaScript for Web",
    subItems: [
      { id: "202-1", title: "JavaScript Basics", path: "/courses/web-development/javascript-basics" },
      { id: "202-2", title: "DOM Manipulation", path: "/courses/web-development/dom-manipulation" },
    ],
  },
  {
    id: "203",
    title: "203: Frontend Frameworks",
    subItems: [
      { id: "203-1", title: "Introduction to React", path: "/courses/web-development/react-intro" },
      { id: "203-2", title: "Vue.js Basics", path: "/courses/web-development/vue-intro" },
      { id: "203-3", title: "Angular Basics", path: "/courses/web-development/angular-intro" },
    ],
  },
  {
    id: "204",
    title: "204: Backend Development",
    subItems: [
      { id: "204-1", title: "Node.js & Express", path: "/courses/web-development/node-express" },
      { id: "204-2", title: "API Development", path: "/courses/web-development/api-development" },
    ],
  },
  {
    id: "205",
    title: "205: Databases & APIs",
    subItems: [
      { id: "205-1", title: "MongoDB Basics", path: "/courses/web-development/mongodb" },
      { id: "205-2", title: "SQL Fundamentals", path: "/courses/web-development/sql-basics" },
      { id: "205-3", title: "REST & GraphQL APIs", path: "/courses/web-development/rest-graphql" },
    ],
  },
];

const WebDevSidebar = ({ theme, sidebarOpen, setSidebarOpen }) => {
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
            Web Development Series
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

export default WebDevSidebar;
