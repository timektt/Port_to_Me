import React, { useEffect, useState } from "react";
import { FaTimes } from "react-icons/fa";
import Footer from "./Footer";

const sidebarItems = [
  { id: "101", title: "Basic Python" },
  { id: "201", title: "Data" },
  { id: "202", title: "Visualization" },
  { id: "203", title: "Data Wrangling & Transform" },
  { id: "204", title: "Statistic Analysis" },
  { id: "205", title: "Statistic Learning" },
];

const Sidebar = ({ theme, activeCourse, sidebarOpen, setSidebarOpen }) => {
  // ✅ ใช้ useState เพื่อให้ Sidebar เปลี่ยนธีมทันที
  const [currentTheme, setCurrentTheme] = useState(theme);

  // ✅ ใช้ useEffect เพื่อให้ Sidebar อัปเดตธีมทุกครั้งที่ `theme` เปลี่ยน
  useEffect(() => {
    setCurrentTheme(theme);
  }, [theme]);

  // ✅ ตรวจจับขนาดหน้าจอ และปิด Sidebar อัตโนมัติเมื่อกลับเป็น Desktop
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth >= 768) {
        setSidebarOpen(false); // ปิด Sidebar เมื่อหน้าจอขยายใหญ่ขึ้น
      }
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (
    <>
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/40 backdrop-blur-sm z-40 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <aside
        className={`fixed top-16 left-0 w-64 h-[calc(100vh-64px)] overflow-y-auto z-40 p-6
        transition-transform duration-300 ease-in-out ${sidebarOpen ? "translate-x-0" : "-translate-x-full"} md:translate-x-0
        ${currentTheme === "dark" ? "bg-gray-900 text-white border-gray-700" : "bg-white text-black border-gray-200"} border-r shadow-lg`}
      >
        {/* Close button สำหรับมือถือ */}
        <button
          className="md:hidden absolute top-4 right-4 text-xl"
          onClick={() => setSidebarOpen(false)}
        >
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
              className={`p-2 rounded-lg cursor-pointer transition ${
                currentTheme === "dark"
                  ? "bg-gray-800 text-white hover:bg-gray-700"
                  : "bg-gray-100 text-black hover:bg-gray-300"
              }`}
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
