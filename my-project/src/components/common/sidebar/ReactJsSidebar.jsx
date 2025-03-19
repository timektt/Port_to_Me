import React, { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { FaTimes, FaChevronDown, FaChevronRight } from "react-icons/fa";

const sidebarItems = [
  // ✅ 101: พื้นฐานของ React.js
  {
    id: "101",
    title: "101: Introduction to React.js",
    subItems: [
      { id: "101-1", title: "What is React.js?", path: "/courses/reactjs-series/intro" },
      { id: "101-2", title: "Setting Up React Project", path: "/courses/reactjs-series/setup" },
      { id: "101-3", title: "JSX & Rendering", path: "/courses/reactjs-series/jsx-rendering" },
      { id: "101-4", title: "React Virtual DOM", path: "/courses/reactjs-series/virtual-dom" },
      { id: "101-5", title: "React vs Other Frameworks", path: "/courses/reactjs-series/react-vs-frameworks" },
    ],
  },

  // ✅ 201: การจัดการ Component & Props
  {
    id: "201",
    title: "201: React Components & Props",
    subItems: [
      { id: "201-1", title: "Functional & Class Components", path: "/courses/reactjs-series/components" },
      { id: "201-2", title: "Props & Prop Drilling", path: "/courses/reactjs-series/props" },
      { id: "201-3", title: "Component Lifecycle Methods", path: "/courses/reactjs-series/lifecycle" },
      { id: "201-4", title: "Reusable Components", path: "/courses/reactjs-series/reusable-components" },
      { id: "201-5", title: "Composition vs Inheritance", path: "/courses/reactjs-series/composition-vs-inheritance" },
    ],
  },

  // ✅ 202: การจัดการ State
  {
    id: "202",
    title: "202: State Management",
    subItems: [
      { id: "202-1", title: "Using useState Hook", path: "/courses/reactjs-series/state" },
      { id: "202-2", title: "React Context API", path: "/courses/reactjs-series/context-api" },
      { id: "202-3", title: "Redux Basics", path: "/courses/reactjs-series/redux" },
      { id: "202-4", title: "Recoil & Zustand", path: "/courses/reactjs-series/recoil-zustand" },
      { id: "202-5", title: "Managing Global State", path: "/courses/reactjs-series/global-state" },
    ],
  },

  // ✅ 203: React Hooks ที่สำคัญ
  {
    id: "203",
    title: "203: React Hooks",
    subItems: [
      { id: "203-1", title: "Introduction to Hooks", path: "/courses/reactjs-series/hooks-intro" },
      { id: "203-2", title: "useEffect & Side Effects", path: "/courses/reactjs-series/useeffect" },
      { id: "203-3", title: "useRef & Manipulating DOM", path: "/courses/reactjs-series/useref" },
      { id: "203-4", title: "useReducer & State Management", path: "/courses/reactjs-series/usereducer" },
      { id: "203-5", title: "Custom Hooks", path: "/courses/reactjs-series/custom-hooks" },
    ],
  },

  // ✅ 204: การจัดการ Routing
  {
    id: "204",
    title: "204: React Router & Navigation",
    subItems: [
      { id: "204-1", title: "Introduction to React Router", path: "/courses/reactjs-series/react-router" },
      { id: "204-2", title: "Nested Routes & Dynamic Routes", path: "/courses/reactjs-series/nested-routes" },
      { id: "204-3", title: "Programmatic Navigation", path: "/courses/reactjs-series/navigation" },
      { id: "204-4", title: "Protected Routes & Authentication", path: "/courses/reactjs-series/protected-routes" },
      { id: "204-5", title: "Lazy Loading with React Router", path: "/courses/reactjs-series/lazy-loading" },
    ],
  },

  // ✅ 205: การเชื่อมต่อ API และการจัดการ Data Fetching
  {
    id: "205",
    title: "205: Fetching Data & API Integration",
    subItems: [
      { id: "205-1", title: "Fetching Data with Fetch API", path: "/courses/reactjs-series/fetch-api" },
      { id: "205-2", title: "Using Axios for HTTP Requests", path: "/courses/reactjs-series/axios" },
      { id: "205-3", title: "Handling Loading & Errors", path: "/courses/reactjs-series/loading-errors" },
      { id: "205-4", title: "GraphQL Integration", path: "/courses/reactjs-series/graphql" },
      { id: "205-5", title: "Caching & Optimizing API Calls", path: "/courses/reactjs-series/caching-api" },
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
          className={`fixed top-16 left-0 w-64 h-[calc(100vh-70px)] overflow-y-auto z-50 p-4 transition-transform duration-300 ease-in-out
            ${sidebarOpen ? "translate-x-0" : "-translate-x-full"} md:translate-x-0
            ${theme === "dark" ? "bg-gray-900 text-white" : "bg-white text-black"} shadow-lg pb-20`} // ✅ เพิ่ม pb-20
        >
          <button className="md:hidden absolute top-4 right-4 text-xl" onClick={() => setSidebarOpen(false)}>
            <FaTimes />
          </button>
    
          <h2 className="text-xl font-bold mb-4">
            <span className={`inline-block px-3 py-1 rounded-md ${theme === "dark" ? "bg-gray-700" : "bg-gray-200"}`}>
              React.js Series
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

export default ReactJsSidebar;
