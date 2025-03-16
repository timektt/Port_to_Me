import React, { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { FaTimes, FaChevronDown, FaChevronRight } from "react-icons/fa";

const sidebarItems = [
  {
    id: "101",
    title: "101: Node.js Basics",
    subItems: [
      { id: "101-1", title: "Introduction to Node.js", path: "/courses/nodejs-series/intro" },
      { id: "101-2", title: "Setting Up Node.js", path: "/courses/nodejs-series/setup" },
      { id: "101-3", title: "Running JavaScript in Node.js", path: "/courses/nodejs-series/running-js" },
      { id: "101-4", title: "Understanding Modules & require()", path: "/courses/nodejs-series/modules" },
      { id: "101-5", title: "Node.js Global Objects & Process", path: "/courses/nodejs-series/global-objects" },
    ],
  },
  {
    id: "201",
    title: "201: Asynchronous JavaScript",
    subItems: [
      { id: "201-1", title: "Understanding Async & Callbacks", path: "/courses/nodejs-series/async" },
      { id: "201-2", title: "Promises & Async/Await", path: "/courses/nodejs-series/promises" },
      { id: "201-3", title: "Handling Errors in Async Code", path: "/courses/nodejs-series/async-error-handling" },
      { id: "201-4", title: "Working with Files Asynchronously", path: "/courses/nodejs-series/async-file" },
    ],
  },
  {
    id: "202",
    title: "202: Event Loop & Async Operations",
    subItems: [
      { id: "202-1", title: "Understanding the Event Loop", path: "/courses/nodejs-series/event-loop" },
      { id: "202-2", title: "Working with Timers & I/O", path: "/courses/nodejs-series/io-operations" },
      { id: "202-3", title: "Event Emitters & Listeners", path: "/courses/nodejs-series/event-emitters" },
      { id: "202-4", title: "Handling Streams in Node.js", path: "/courses/nodejs-series/node-streams" },
    ],
  },
  {
    id: "203",
    title: "203: API Development",
    subItems: [
      { id: "203-1", title: "Building RESTful APIs", path: "/courses/nodejs-series/rest-api" },
      { id: "203-2", title: "Handling HTTP Requests", path: "/courses/nodejs-series/http-requests" },
      { id: "203-3", title: "Working with Middleware", path: "/courses/nodejs-series/middleware" },
      { id: "203-4", title: "Data Validation & Error Handling", path: "/courses/nodejs-series/api-validation" },
      { id: "203-5", title: "Implementing Authentication", path: "/courses/nodejs-series/api-authentication" },
    ],
  },
  {
    id: "204",
    title: "204: Express.js Framework",
    subItems: [
      { id: "204-1", title: "Introduction to Express.js", path: "/courses/nodejs-series/express-intro" },
      { id: "204-2", title: "Routing in Express.js", path: "/courses/nodejs-series/express-routing" },
      { id: "204-3", title: "Handling Middleware in Express", path: "/courses/nodejs-series/express-middleware" },
      { id: "204-4", title: "Serving Static Files", path: "/courses/nodejs-series/express-static-files" },
      { id: "204-5", title: "Express.js & CORS", path: "/courses/nodejs-series/express-cors" },
    ],
  },
  {
    id: "205",
    title: "205: Database Integration",
    subItems: [
      { id: "205-1", title: "Connecting to MongoDB", path: "/courses/nodejs-series/mongodb" },
      { id: "205-2", title: "Using PostgreSQL with Node.js", path: "/courses/nodejs-series/postgresql" },
      { id: "205-3", title: "Working with Mongoose", path: "/courses/nodejs-series/mongoose" },
      { id: "205-4", title: "Sequelize for SQL Databases", path: "/courses/nodejs-series/sequelize" },
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
        className={`fixed top-16 left-0 w-64 h-[calc(100vh-64px)] overflow-y-auto z-50 p-4 transition-transform duration-300 ease-in-out
          ${sidebarOpen ? "translate-x-0" : "-translate-x-full"} md:translate-x-0
          ${theme === "dark" ? "bg-gray-900 text-white" : "bg-white text-black"} shadow-lg`}
      >
        <button className="md:hidden absolute top-4 right-4 text-xl" onClick={() => setSidebarOpen(false)}>
          <FaTimes />
        </button>

        <h2 className="text-xl font-bold mb-4">
          <span className={`inline-block px-3 py-1 rounded-md ${theme === "dark" ? "bg-gray-700" : "bg-gray-200"}`}>
            Node.js Series
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

export default NodeJsSidebar;
