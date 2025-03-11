import React, { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { FaTimes, FaChevronDown, FaChevronRight, FaArrowLeft, FaSun, FaMoon } from "react-icons/fa";

const sidebarItems = [
  {
    id: "101",
    title: "101: Introduction to Programming",
    subItems: [
      { id: "101-1", title: "What is Programming?", path: "/courses/basic-programming/intro" },
      { id: "101-2", title: "Programming Languages Overview", path: "/courses/basic-programming/languages" },
    ],
  },
  {
    id: "201",
    title: "201: Variables & Data Types",
    subItems: [
      { id: "201-1", title: "Understanding Variables", path: "/courses/basic-programming/variables" },
      { id: "201-2", title: "Primitive Data Types", path: "/courses/basic-programming/primitive-types" },
      { id: "201-3", title: "Advanced Data Types", path: "/courses/basic-programming/advanced-types" },
    ],
  },
  {
    id: "202",
    title: "202: Control Flow & Loops",
    subItems: [
      { id: "202-1", title: "If-Else Statements", path: "/courses/basic-programming/if-else" },
      { id: "202-2", title: "Loops & Iteration", path: "/courses/basic-programming/loops" },
      { id: "202-3", title: "Switch Case", path: "/courses/basic-programming/switch-case" },
    ],
  },
  {
    id: "203",
    title: "203: Functions & Modules",
    subItems: [
      { id: "203-1", title: "Functions in Programming", path: "/courses/basic-programming/functions" },
      { id: "203-2", title: "Modules & Imports", path: "/courses/basic-programming/modules" },
    ],
  },
  {
    id: "204",
    title: "204: Object-Oriented Programming (OOP)",
    subItems: [
      { id: "204-1", title: "Introduction to OOP", path: "/courses/basic-programming/oop" },
      { id: "204-2", title: "Classes & Objects", path: "/courses/basic-programming/classes" },
      { id: "204-3", title: "Encapsulation & Inheritance", path: "/courses/basic-programming/encapsulation-inheritance" },
      { id: "204-4", title: "Polymorphism & Abstraction", path: "/courses/basic-programming/polymorphism" },
    ],
  },
  {
    id: "205",
    title: "205: Debugging & Error Handling",
    subItems: [
      { id: "205-1", title: "Common Programming Errors", path: "/courses/basic-programming/errors" },
      { id: "205-2", title: "Debugging Techniques", path: "/courses/basic-programming/debugging" },
      { id: "205-3", title: "Exception Handling", path: "/courses/basic-programming/exception-handling" },
    ],
  },
];

const BasicProgrammingMobileMenu = ({ onClose, theme, setTheme }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const [expandedSections, setExpandedSections] = useState({});

  const toggleSection = (id) => {
    setExpandedSections((prev) => ({
      ...prev,
      [id]: !prev[id],
    }));
  };

  const toggleTheme = () => {
    const newTheme = theme === "dark" ? "light" : "dark";
    setTheme(newTheme);
    localStorage.setItem("theme", newTheme);
  };

  return (
    <div className={`fixed top-0 left-0 w-64 h-full p-4 z-50 shadow-lg transition-all duration-300 
      ${theme === "dark" ? "bg-gray-900 text-white" : "bg-white text-black"}`}>

      {/* ✅ ปุ่มปิดเมนู (X) */}
      <button 
        className={`absolute right-4 top-4 text-2xl transition-colors duration-200 
          ${theme === "dark" ? "text-white hover:text-gray-400" : "text-black hover:text-gray-600"}`}
        onClick={onClose}
      >
        <FaTimes />
      </button>

      {/* ✅ โลโก้ + ปุ่ม Dark/Light Mode */}
      <div className="mt-6 flex items-center justify-between mb-3">
        <span className="text-lg font-bold cursor-pointer hover:text-gray-400 transition">
          Basic Programming
        </span>
        <button className="cursor-pointer transition-transform transform hover:scale-110" onClick={toggleTheme}>
          {theme === "dark" ? <FaSun className="text-yellow-400 text-xl" /> : <FaMoon className="text-blue-400 text-xl" />}
        </button>
      </div>

      {/* ✅ ปุ่ม Back to Main Menu */}
      <button 
        className="flex items-center text-sm text-gray-400 hover:text-gray-300 mb-3 transition"
        onClick={onClose}
      >
        <FaArrowLeft className="mr-2" /> Back to main menu
      </button>

      {/* ✅ รายการบทเรียน (Dropdown) */}
      <ul className="space-y-2 mt-4">
        {sidebarItems.map((item) => (
          <li key={item.id} className="border-b border-gray-700">
            {/* ปุ่มกดขยายหมวดหมู่ */}
            <button
              className="flex items-center justify-between w-full p-3 rounded-lg transition duration-300 ease-in-out
                hover:bg-gray-700 hover:shadow-lg text-left"
              onClick={() => toggleSection(item.id)}
            >
              {item.title}
              {expandedSections[item.id] ? <FaChevronDown /> : <FaChevronRight />}
            </button>

            {/* ✅ แสดงหมวดหมู่ย่อย */}
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

export default BasicProgrammingMobileMenu;
