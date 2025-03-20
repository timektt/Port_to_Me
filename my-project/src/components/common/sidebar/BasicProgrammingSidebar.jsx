import React, { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { FaTimes, FaChevronDown, FaChevronRight } from "react-icons/fa";

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

const BasicProgrammingSidebar = ({ theme, sidebarOpen, setSidebarOpen }) => {
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
             BasicProgramming
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
export default BasicProgrammingSidebar;
