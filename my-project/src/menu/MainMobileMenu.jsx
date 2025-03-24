import React from "react";
import {
  FaTimes,
  FaChevronRight,
  FaArrowLeft,
  FaSun,
  FaMoon,
} from "react-icons/fa";
import { useNavigate } from "react-router-dom";

const MainMobileMenu = ({ onClose, theme, setTheme }) => {
  const navigate = useNavigate();

  const toggleTheme = () => {
    const newTheme = theme === "dark" ? "light" : "dark";
    setTheme(newTheme);
    localStorage.setItem("theme", newTheme);
  };

  return (
    <>
      {/* ✅ Overlay ด้านหลัง */}
      <div
        className="fixed inset-0 bg-black bg-opacity-30 z-40"
        onClick={onClose}
        aria-label="Close menu overlay"
      ></div>

      {/* ✅ เมนูหลัก */}
      <div
        className={`fixed top-0 left-0 w-64 h-full p-4 z-50 shadow-lg transition-all duration-300
          ${theme === "dark" ? "bg-gray-900 text-white" : "bg-white text-black"}`}
      >
        {/* ✅ ปุ่มปิดเมนู (X) */}
        <button
          role="button"
          aria-label="Close menu"
          className={`absolute right-4 top-4 text-2xl transition-colors duration-200
            ${theme === "dark" ? "text-white hover:text-gray-400" : "text-black hover:text-gray-600"}`}
          onClick={onClose}
        >
          <FaTimes />
        </button>

        {/* ✅ โลโก้ + Superbear + ปุ่ม Toggle Theme */}
        <div className="mt-6 flex items-center mb-3">
          <img
            src="/spm2.jpg"
            alt="Logo"
            className="w-8 h-8 mr-2 object-cover rounded-full"
          />
          <div className="flex items-center space-x-2">
            <span className="text-lg font-bold cursor-pointer hover:text-gray-400 transition">
              Superbear
            </span>
            <button
              onClick={toggleTheme}
              tabIndex="0"
              className="cursor-pointer transition-transform transform hover:scale-110"
              aria-label="Toggle dark mode"
            >
              {theme === "dark" ? (
                <FaSun className="text-yellow-400 text-2xl" />
              ) : (
                <FaMoon className="text-blue-400 text-2xl" />
              )}
            </button>
          </div>
        </div>

        {/* ✅ ปุ่มกลับ */}
        <button
          className="flex items-center text-sm text-gray-400 hover:text-gray-300 mb-3 transition"
          onClick={onClose}
        >
          <FaArrowLeft className="mr-2" /> กลับไปที่เมนูหลัก
        </button>

        {/* ✅ รายการเมนู */}
        <ul className="mt-3 space-y-3">
          <li>
            <button
              onClick={() => {
                navigate("/courses");
                onClose();
              }}
              className="w-full flex justify-between items-center text-left hover:text-gray-300 transition"
            >
              Courses <FaChevronRight />
            </button>
          </li>
          <li>
            <button
              onClick={() => {
                navigate("/tags");
                onClose();
              }}
              className="w-full flex justify-between items-center text-left hover:text-gray-300 transition"
            >
              Tags <FaChevronRight />
            </button>
          </li>
          <li>
            <button
              onClick={() => {
                navigate("/about");
                onClose();
              }}
              className="w-full flex justify-between items-center text-left hover:text-gray-300 transition"
            >
              Contact us <FaChevronRight />
            </button>
          </li>
        </ul>
      </div>
    </>
  );
};

export default MainMobileMenu;
