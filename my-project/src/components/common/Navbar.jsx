import React, { useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom"; // ✅ ใช้ useLocation เพื่อตรวจสอบ path ปัจจุบัน
import ProfileInfo from "../ProfileInfo";
import { FaYoutube, FaFacebook, FaGithub, FaBars, FaTimes, FaSearch, FaSun, FaMoon } from "react-icons/fa";
import MainMobileMenu from "../MainMobileMenu"; // ✅ ใช้สำหรับหน้า Home
import PythonMobileMenu from "../PythonMobileMenu"; // ✅ ใช้สำหรับหน้า PythonSeries

const Navbar = ({ theme, setTheme, onMenuToggle }) => {
  const [menuOpen, setMenuOpen] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const navigate = useNavigate();
  const location = useLocation(); // ✅ ตรวจสอบ path ปัจจุบัน

  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth >= 988) {
        setMenuOpen(false);
        setMobileMenuOpen(false);
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
        <button className="md:hidden text-3xl" onClick={() => setMobileMenuOpen(true)}>
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

      {/* ✅ Mobile Menu (ตรวจสอบ path ว่าควรใช้ Main หรือ Python) */}
      {mobileMenuOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50">
          {location.pathname.startsWith("/courses/python-series") ? (
            <PythonMobileMenu onClose={() => setMobileMenuOpen(false)} theme={theme} setTheme={setTheme} />
          ) : (
            <MainMobileMenu onClose={() => setMobileMenuOpen(false)} theme={theme} setTheme={setTheme} />
          )}
        </div>
      )}
    </nav>
  );
};

export default Navbar;
