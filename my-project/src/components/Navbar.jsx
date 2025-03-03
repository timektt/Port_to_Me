import React, { useState, useEffect } from "react";
import ProfileInfo from "./ProfileInfo";
import { FaYoutube, FaFacebook, FaGithub, FaBars, FaTimes, FaSearch } from "react-icons/fa";

const Navbar = () => {
  const [menuOpen, setMenuOpen] = useState(false);

  // **📌 ปิดเมนูอัตโนมัติเมื่อหน้าจอขยายเกิน 988px**
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth >= 988) {
        setMenuOpen(false);
      }
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (
    <nav className="bg-gray-800 text-white px-4 py-2 flex justify-between items-center text-base md:text-lg relative">
      {/* **Left: Hamburger Menu + Profile** */}
      <div className="flex items-center gap-3">
        {/* **📌 Hamburger Button (แสดงเฉพาะเมื่อจอเล็ก)** */}
        <button className="md:hidden text-white text-3xl" onClick={() => setMenuOpen(true)}>
          <FaBars />
        </button>
        <ProfileInfo />
      </div>

      {/* **Right: Social Links + Search Box (แสดงเฉพาะจอใหญ่)** */}
      <div className="hidden md:flex items-center gap-6">
        <a href="https://youtube.com" target="_blank" rel="noopener noreferrer">
          <FaYoutube className="text-2xl md:text-3xl hover:text-red-500" />
        </a>
        <a href="https://facebook.com" target="_blank" rel="noopener noreferrer">
          <FaFacebook className="text-2xl md:text-3xl hover:text-blue-500" />
        </a>
        <a href="https://github.com" target="_blank" rel="noopener noreferrer">
          <FaGithub className="text-2xl md:text-3xl hover:text-gray-500" />
        </a>
        <div className="relative w-32 md:w-48">
          <input
            type="text"
            placeholder="Search..."
            className="p-2 pl-8 w-full rounded-lg bg-gray-700 text-white border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <FaSearch className="absolute left-2 top-2.5 text-gray-400" />
        </div>
      </div>

      {/* **📌 Fullscreen Sidebar Menu (เมื่อ Hamburger เปิด) ** */}
      {menuOpen && (
        <>
          {/* **📌 Sidebar Container (เฉพาะ Sidebar เท่านั้นที่ทึบ)** */}
          <div className="fixed top-0 left-0 w-64 h-full bg-gray-900 text-white shadow-lg z-50 p-6">
            {/* **📌 ปุ่มปิดเมนู** */}
            <button className="self-end text-3xl absolute right-4 top-4" onClick={() => setMenuOpen(false)}>
              <FaTimes />
            </button>

            {/* **📌 Menu Links** */}
            <div className="mt-6 flex flex-col gap-6 text-lg">
              <a href="#" className="hover:text-gray-400">Courses</a>
              <a href="#" className="hover:text-gray-400">Posts</a>
              <a href="https://youtube.com" target="_blank" rel="noopener noreferrer" className="hover:text-gray-400">
                Youtube
              </a>
              <a href="https://facebook.com" target="_blank" rel="noopener noreferrer" className="hover:text-gray-400">
                Facebook
              </a>
              <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="hover:text-gray-400">
                Github
              </a>
            </div>
          </div>

          {/* **📌 Overlay คลุมเฉพาะ Sidebar เท่านั้น (ทำให้เมนูเด่นขึ้นแต่ไม่บังเนื้อหา)** */}
          <div
            className="fixed inset-0 backdrop-blur-sm bg-black/20 z-40"
            onClick={() => setMenuOpen(false)}
          ></div>
        </>
      )}
    </nav>
  );
};

export default Navbar;
