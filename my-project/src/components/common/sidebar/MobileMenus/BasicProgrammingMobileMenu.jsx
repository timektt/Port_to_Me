import React, { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { FaTimes, FaChevronDown, FaChevronRight, FaArrowLeft, FaSun, FaMoon } from "react-icons/fa";

const sidebarItems = [
  {
    id: "101",
    title: "101: Introduction to Programming",
    subItems: [
      { id: "101-1", title: "What is Programming?", path: "/courses/basic-programming/intro" },
      { id: "101-2", title: "How Computers Execute Code", path: "/courses/basic-programming/computer-execution" },
      { id: "101-3", title: "Types of Programming Languages", path: "/courses/basic-programming/programming-languages" },
      { id: "101-4", title: "Compilers vs Interpreters", path: "/courses/basic-programming/compilers-interpreters" },
      { id: "101-5", title: "Setting Up a Programming Environment", path: "/courses/basic-programming/setup" },
    ],
  },
  {
    id: "201",
    title: "201: Variables & Data Types",
    subItems: [
      { id: "201-1", title: "Understanding Variables", path: "/courses/basic-programming/variables" },
      { id: "201-2", title: "Primitive Data Types", path: "/courses/basic-programming/data-types" },
      { id: "201-3", title: "Type Conversion & Casting", path: "/courses/basic-programming/type-conversion" },
      { id: "201-4", title: "Constants & Immutable Data", path: "/courses/basic-programming/constants" },
      { id: "201-5", title: "Scope & Lifetime of Variables", path: "/courses/basic-programming/scope" },
    ],
  },
  {
    id: "202",
    title: "202: Control Flow & Loops",
    subItems: [
      { id: "202-1", title: "Conditional Statements", path: "/courses/basic-programming/conditions" },
      { id: "202-2", title: "Loops: for & while", path: "/courses/basic-programming/loops" },
      { id: "202-3", title: "Break & Continue Statements", path: "/courses/basic-programming/break-continue" },
      { id: "202-4", title: "Nested Loops", path: "/courses/basic-programming/nested-loops" },
      { id: "202-5", title: "Recursion Basics", path: "/courses/basic-programming/recursion" },
    ],
  },
  {
    id: "203",
    title: "203: Functions & Modules",
    subItems: [
      { id: "203-1", title: "Defining Functions", path: "/courses/basic-programming/functions" },
      { id: "203-2", title: "Working with Modules", path: "/courses/basic-programming/modules" },
      { id: "203-3", title: "Function Parameters & Arguments", path: "/courses/basic-programming/parameters" },
      { id: "203-4", title: "Return Values & Scope", path: "/courses/basic-programming/return-values" },
      { id: "203-5", title: "Lambda Functions & Anonymous Functions", path: "/courses/basic-programming/lambda-functions" },
    ],
  },
  {
    id: "204",
    title: "204: Object-Oriented Programming (OOP)",
    subItems: [
      { id: "204-1", title: "Classes & Objects", path: "/courses/basic-programming/oop" },
      { id: "204-2", title: "Encapsulation & Inheritance", path: "/courses/basic-programming/oop-inheritance" },
      { id: "204-3", title: "Polymorphism & Method Overriding", path: "/courses/basic-programming/polymorphism" },
      { id: "204-4", title: "Abstraction & Interfaces", path: "/courses/basic-programming/abstraction" },
      { id: "204-5", title: "OOP Design Patterns", path: "/courses/basic-programming/oop-design-patterns" },
    ],
  },
  {
    id: "205",
    title: "205: Debugging & Error Handling",
    subItems: [
      { id: "205-1", title: "Common Programming Errors", path: "/courses/basic-programming/debugging" },
      { id: "205-2", title: "Using Debugging Tools", path: "/courses/basic-programming/debugging-tools" },
      { id: "205-3", title: "Types of Errors (Syntax, Runtime, Logic)", path: "/courses/basic-programming/error-types" },
      { id: "205-4", title: "Exception Handling", path: "/courses/basic-programming/exception-handling" },
      { id: "205-5", title: "Logging & Performance Monitoring", path: "/courses/basic-programming/logging-monitoring" },
    ],
  },
];
const BasicProgrammingMobileMenu = ({ onClose, theme, setTheme }) => {
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
 

export default BasicProgrammingMobileMenu;
