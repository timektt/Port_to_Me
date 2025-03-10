import React from "react";
import { FaTimes } from "react-icons/fa";

const sidebarItems = [
  { id: "101", title: "Introduction to Programming" },
  { id: "201", title: "Variables & Data Types" },
  { id: "202", title: "Control Flow & Loops" },
  { id: "203", title: "Functions & Modules" },
  { id: "204", title: "Object-Oriented Programming (OOP)" },
  { id: "205", title: "Debugging & Error Handling" },
];

const BasicProgrammingSidebar = ({ theme, sidebarOpen, setSidebarOpen }) => {
  return (
    <>
      {/* ✅ Overlay สำหรับ Mobile */}
      {sidebarOpen && (
        <div className="fixed inset-0 bg-black/40 backdrop-blur-sm z-40 md:hidden" onClick={() => setSidebarOpen(false)} />
      )}

      {/* ✅ Sidebar (ใช้ได้ทั้ง Mobile & Desktop) */}
      <aside className={`fixed top-16 left-0 w-64 h-[calc(100vh-64px)] overflow-y-auto z-50 p-6 transition-transform duration-300 ease-in-out
          ${sidebarOpen ? "translate-x-0" : "-translate-x-full"} md:translate-x-0
          ${theme === "dark" ? "bg-gray-900 text-white" : "bg-white text-black"} shadow-lg`}>
        
        {/* ✅ ปุ่มปิด Sidebar บนมือถือ */}
        <button className="md:hidden absolute top-4 right-4 text-xl" onClick={() => setSidebarOpen(false)}>
          <FaTimes />
        </button>

        {/* ✅ Header */}
        <h2 className="text-xl font-bold mb-4">
          <span className={`inline-block px-3 py-1 rounded-md ${theme === "dark" ? "bg-gray-700" : "bg-gray-200"}`}>
            Basic Programming
          </span>
        </h2>

        {/* ✅ Sidebar items */}
        <ul className="space-y-2 mt-4">
          {sidebarItems.map((item) => (
            <li
              key={item.id}
              className={`p-3 rounded-lg cursor-pointer transition duration-300 ease-in-out
                ${theme === "dark" ? "text-white hover:bg-gray-700" : "text-black hover:bg-gray-200"}
                hover:shadow-lg`}>
              {item.title}
            </li>
          ))}
        </ul>
      </aside>
    </>
  );
};

export default BasicProgrammingSidebar;
