import React, { useEffect, useState } from "react";
import { FaTimes } from "react-icons/fa";

const Sidebar = ({ theme, activeCourse, sidebarItems, sidebarOpen, setSidebarOpen }) => {
  const [currentTheme, setCurrentTheme] = useState(theme);
  const [hoveredItem, setHoveredItem] = useState(null);
  const [selectedItem, setSelectedItem] = useState(null);

  useEffect(() => {
    setCurrentTheme(theme);
  }, [theme]);

  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth >= 768 && sidebarOpen) {
        setSidebarOpen(false);
      }
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [sidebarOpen, setSidebarOpen]);

  // เมื่อคลิกพื้นที่ว่าง ให้ล้าง `selectedItem`
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (!event.target.closest(".sidebar-item")) {
        setSelectedItem(null);
      }
    };
    document.addEventListener("click", handleClickOutside);
    return () => document.removeEventListener("click", handleClickOutside);
  }, []);

  return (
    <>
      {sidebarOpen && (
        <div className="fixed inset-0 bg-black/40 backdrop-blur-sm z-40 md:hidden" onClick={() => setSidebarOpen(false)} />
      )}

      <aside
        className={`fixed top-16 left-0 w-64 h-[calc(100vh-64px)] overflow-y-auto z-40 p-6
        transition-transform duration-300 ease-in-out ${sidebarOpen ? "translate-x-0" : "-translate-x-full"} md:translate-x-0
        ${currentTheme === "dark" ? "bg-gray-900 text-white border-gray-700" : "bg-white text-black border-gray-200"} border-r shadow-lg`}
      >
        {/* ปุ่มปิด Sidebar บนมือถือ */}
        <button className="md:hidden absolute top-4 right-4 text-xl" onClick={() => setSidebarOpen(false)}>
          <FaTimes />
        </button>

        {/* Header */}
        <h2 className="text-xl font-bold mb-4">
          <span className={`inline-block px-3 py-1 rounded-md ${currentTheme === "dark" ? "bg-gray-700" : "bg-gray-200"}`}>
            {activeCourse}
          </span>
        </h2>

        {/* Sidebar items */}
        <ul className="space-y-2 mt-4">
          {sidebarItems.map((item) => (
            <li
              key={item.id}
              className={`p-2 rounded-lg cursor-pointer transition sidebar-item ${
                currentTheme === "dark"
                  ? selectedItem === item.id
                    ? "bg-gray-700 text-white"
                    : hoveredItem === item.id
                    ? "bg-gray-600 text-white"
                    : "text-white"
                  : selectedItem === item.id
                  ? "bg-gray-300 text-black"
                  : hoveredItem === item.id
                  ? "bg-gray-200 text-black"
                  : "text-black"
              }`}
              tabIndex={0}
              onMouseEnter={() => setHoveredItem(item.id)}
              onMouseLeave={() => setHoveredItem(null)}
              onClick={() => {
                setSelectedItem(item.id);
                setHoveredItem(null);
              }}
              onBlur={() => setSelectedItem(null)}
            >
              {item.title}
            </li>
          ))}
        </ul>
      </aside>
    </>
  );
};

export default Sidebar;
