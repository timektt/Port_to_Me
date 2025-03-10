import React from "react";
import { FaTimes } from "react-icons/fa";

const defaultSidebarItems = [
  { id: "101", title: "Basic Python" },
  { id: "201", title: "Data" },
  { id: "202", title: "Visualization" },
  { id: "203", title: "Data Wrangling & Transform" },
  { id: "204", title: "Statistic Analysis" },
  { id: "205", title: "Statistic Learning" },
];

const PythonSidebar = ({ theme, sidebarOpen, setSidebarOpen, sidebarItems = defaultSidebarItems }) => {
  return (
    <>
      {/* Overlay สำหรับ Mobile */}
      {sidebarOpen && (
        <div className="fixed inset-0 bg-black/40 backdrop-blur-sm z-40 md:hidden" onClick={() => setSidebarOpen(false)} />
      )}

      {/* Sidebar (ใช้ได้ทั้ง Mobile & Desktop) */}
      <aside
        className={`fixed top-16 left-0 w-64 h-[calc(100vh-64px)] overflow-y-auto z-50 p-6 transition-transform duration-300 ease-in-out
          ${sidebarOpen ? "translate-x-0" : "-translate-x-full"} md:translate-x-0
          ${theme === "dark" ? "bg-gray-900 text-white" : "bg-white text-black"} shadow-lg`}
      >
        {/* ปุ่มปิด Sidebar บนมือถือ */}
        <button className="md:hidden absolute top-4 right-4 text-xl" onClick={() => setSidebarOpen(false)}>
          <FaTimes />
        </button>

        {/* Header */}
        <h2 className="text-xl font-bold mb-4">
          <span className={`inline-block px-3 py-1 rounded-md ${theme === "dark" ? "bg-gray-700" : "bg-gray-200"}`}>
            Python Series
          </span>
        </h2>

        {/* Sidebar items */}
        <ul className="space-y-2 mt-4">
          {sidebarItems.length > 0 ? (
            sidebarItems.map((item) => (
              <li
                key={item.id}
                className={`p-3 rounded-lg cursor-pointer transition duration-300 ease-in-out
                  ${theme === "dark" ? "text-white hover:bg-gray-700" : "text-black hover:bg-gray-200"}
                  hover:shadow-lg`}
              >
                {item.title}
              </li>
            ))
          ) : (
            <li className="text-gray-500 text-sm italic">No items available</li>
          )}
        </ul>
      </aside>
    </>
  );
};

export default PythonSidebar;
