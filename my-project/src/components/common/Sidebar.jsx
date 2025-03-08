import React, { useEffect, useState } from "react";
import { FaTimes } from "react-icons/fa";

const sidebarItems = [
  { id: "101", title: "Basic Python" },
  { id: "201", title: "Data" },
  { id: "202", title: "Visualization" },
  { id: "203", title: "Data Wrangling & Transform" },
  { id: "204", title: "Statistic Analysis" },
  { id: "205", title: "Statistic Learning" },
];

const Sidebar = ({ theme, activeCourse, sidebarOpen, setSidebarOpen }) => {
  const [currentTheme, setCurrentTheme] = useState(theme);
  const [hoveredItem, setHoveredItem] = useState(null);
  const [selectedItem, setSelectedItem] = useState(null); // ✅ ใช้สำหรับไอเท็มที่ถูกเลือก

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

  // ✅ เมื่อคลิกที่พื้นที่ว่าง ให้ล้าง `selectedItem`
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

      <aside className={`fixed top-16 left-0 w-64 h-[calc(100vh-64px)] overflow-y-auto z-40 p-6 
        transition-transform duration-300 ease-in-out ${sidebarOpen ? "translate-x-0" : "-translate-x-full"} md:translate-x-0
        ${currentTheme === "dark" ? "bg-gray-900 text-white border-gray-700" : "bg-white text-black border-gray-200"} border-r shadow-lg`}>
        
        {/* Close button สำหรับมือถือ */}
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
                    ? "bg-gray-700 text-white" // ✅ ค้างสีไว้เมื่อเลือก
                    : hoveredItem === item.id
                    ? "bg-gray-600 text-white"
                    : "text-white"
                  : selectedItem === item.id
                    ? "bg-gray-300 text-black" // ✅ ค้างสีไว้เมื่อเลือก
                    : hoveredItem === item.id
                    ? "bg-gray-200 text-black"
                    : "text-black"
              }`}
              tabIndex={0} // ✅ ใช้เพื่อให้ `onBlur` ทำงาน
              onMouseEnter={() => setHoveredItem(item.id)} // ✅ เมาส์เข้า
              onMouseLeave={() => setHoveredItem(null)} // ✅ เมาส์ออก
              onClick={() => {
                setSelectedItem(item.id); // ✅ ค้างสีไว้เมื่อเลือก
                setHoveredItem(null); // ✅ รีเซ็ต hover เมื่อคลิก
              }}
              onBlur={() => setSelectedItem(null)} // ✅ เมื่อคลิกพื้นที่ว่าง ให้ล้างค่า
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
