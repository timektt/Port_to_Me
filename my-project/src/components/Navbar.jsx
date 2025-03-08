import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import ProfileInfo from "./ProfileInfo";
import { FaYoutube, FaFacebook, FaGithub, FaBars, FaTimes, FaSearch, FaSun, FaMoon } from "react-icons/fa";

const Navbar = ({ theme, setTheme, onMenuToggle }) => {
  const [menuOpen, setMenuOpen] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth >= 988) {
        setMenuOpen(false);
      }
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const toggleTheme = () => {
    setTheme(theme === "dark" ? "light" : "dark");
  };

  return (
    <nav className={`navbar px-4 py-2 flex justify-between items-center text-base md:text-lg relative ${theme === "dark" ? "bg-gray-800 text-white" : "bg-white text-gray-900"}`}>
      {/* ✅ Left Section: Profile + Hamburger Menu */}
      <div className="flex items-center gap-3">
        <button className="md:hidden text-3xl" onClick={() => { setMenuOpen(!menuOpen); onMenuToggle && onMenuToggle(); }}>
          <FaBars />
        </button>
        <button onClick={() => navigate("/")} className="flex items-center">
          <ProfileInfo />
        </button>
      </div>

      {/* ✅ Right Section: Social Links + Dark Mode Toggle + Search */}
      <div className="hidden md:flex items-center gap-6">
        <a href="https://youtube.com" target="_blank" rel="noopener noreferrer">
          <FaYoutube className="text-2xl hover:text-red-500" />
        </a>
        <a href="https://facebook.com" target="_blank" rel="noopener noreferrer">
          <FaFacebook className="text-2xl hover:text-blue-500" />
        </a>
        <a href="https://github.com" target="_blank" rel="noopener noreferrer">
          <FaGithub className="text-2xl hover:text-gray-500" />
        </a>

        {/* ✅ Toggle Dark/Light Mode */}
        <button onClick={toggleTheme} className="w-10 h-10 flex items-center justify-center rounded-full transition bg-gray-700 hover:bg-gray-600">
          {theme === "dark" ? <FaSun className="text-yellow-400 text-2xl" /> : <FaMoon className="text-blue-400 text-2xl" />}
        </button>

        {/* ✅ Search Box */}
        <div className="relative w-30 md:w-48">
          <input type="text" placeholder="Search..." className={`p-2 pl-8 w-full rounded-lg border focus:outline-none focus:ring-2 focus:ring-blue-500 ${theme === "dark" ? "bg-gray-700 text-white border-gray-600" : "bg-gray-200 text-gray-900 border-gray-400"}`} />
          <FaSearch className="absolute left-2 top-2.5 text-gray-400" />
        </div>
      </div>

      {/* ✅ Hamburger Sidebar Menu */}
      {menuOpen && (
        <>
          <div className="fixed top-0 left-0 w-64 h-full bg-gray-900 text-white shadow-lg z-50 p-6">
            <button className="self-end text-3xl absolute right-4 top-4" onClick={() => setMenuOpen(false)}>
              <FaTimes />
            </button>

            <div className="mt-6 flex flex-col gap-6 text-lg">
              <button onClick={() => { navigate("/"); setMenuOpen(false); }} className="hover:text-gray-400 text-left">
                Courses
              </button>
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

          {/* ✅ Overlay ปิด Sidebar เมื่อคลิกที่พื้นหลัง */}
          <div className="fixed inset-0 backdrop-blur-sm bg-black/20 z-40" onClick={() => setMenuOpen(false)}></div>
        </>
      )}
    </nav>
  );
};

export default Navbar;
