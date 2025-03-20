import React, { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom"; // ✅ รวมให้อยู่ในบรรทัดเดียว
import { FaTimes, FaChevronDown, FaChevronRight, FaArrowLeft, FaSun, FaMoon } from "react-icons/fa";


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




const ReactJsMobileMenu = ({ onClose, theme, setTheme }) => {
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
    <div className={`fixed top-0 left-0 w-64 h-full p-4 z-50 shadow-lg transition-all duration-300 
      ${theme === "dark" ? "bg-gray-900 text-white" : "bg-white text-black"} overflow-y-auto pb-20`}>

      {/* ✅ ปุ่มปิดเมนู (X) */}
      <button 
        className={`absolute right-4 top-4 text-2xl transition-colors duration-200 
          ${theme === "dark" ? "text-white hover:text-gray-400" : "text-black hover:text-gray-600"}`}
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
            {theme === "dark" ? <FaSun className="text-yellow-400 text-2xl" /> : <FaMoon className="text-blue-400 text-2xl" />}
          </button>
        </div>
      </div>

      {/* ✅ ปุ่ม Back ไปเมนูหลัก */}
      <button 
        className="flex items-center text-sm text-gray-400 hover:text-gray-300 mb-3 transition"
        onClick={onClose}
      >
        <FaArrowLeft className="mr-2" /> กลับไปที่เมนูหลัก
      </button>

      {/* ✅ รายการบทเรียน (Dropdown) */}
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
                    onClick={() => { navigate(subItem.path); onClose(); }}
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

export default ReactJsMobileMenu;
