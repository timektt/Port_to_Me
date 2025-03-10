import React from "react";
import { FaTimes } from "react-icons/fa";

const sidebarItems = [
  { id: "101", title: "Introduction to React.js" },
  { id: "201", title: "React Components & Props" },
  { id: "202", title: "State Management" },
];

const ReactJsSidebar = ({ theme, sidebarOpen, setSidebarOpen }) => {
  return (
    <>
      <aside className={`fixed top-16 left-0 w-64 h-[calc(100vh-64px)] overflow-y-auto z-50 p-6 
        ${sidebarOpen ? "translate-x-0" : "-translate-x-full"} md:translate-x-0
        ${theme === "dark" ? "bg-gray-900 text-white" : "bg-white text-black"} shadow-lg`}>
        <button className="md:hidden absolute top-4 right-4 text-xl" onClick={() => setSidebarOpen(false)}>
          <FaTimes />
        </button>
        <h2 className="text-xl font-bold mb-4">React.js Series</h2>
        <ul className="space-y-2">
          {sidebarItems.map((item) => (
            <li key={item.id} className="p-3 hover:bg-gray-700 rounded-lg cursor-pointer">{item.title}</li>
          ))}
        </ul>
      </aside>
    </>
  );
};

export default ReactJsSidebar;
