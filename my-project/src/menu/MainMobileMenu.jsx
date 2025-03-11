import React from "react";
import { FaTimes, FaChevronRight, FaArrowLeft, FaSun, FaMoon } from "react-icons/fa"; // ✅ Import ไอคอนที่ขาดหายไป
import { useNavigate } from "react-router-dom";

const MainMobileMenu = ({ onClose, theme, setTheme }) => {
  const navigate = useNavigate();

  const toggleTheme = () => {
    const newTheme = theme === "dark" ? "light" : "dark";
    setTheme(newTheme);
    localStorage.setItem("theme", newTheme);
  };

  return (
    <div className={`fixed top-0 left-0 w-64 h-full p-4 z-50 shadow-lg transition-all duration-300 
      ${theme === "dark" ? "bg-gray-900 text-white" : "bg-white text-black"}`}>

      {/* ✅ ปุ่มปิดเมนู (X) */} 
      <button 
        className={`absolute right-4 top-4 text-2xl transition-colors duration-200 
          ${theme === "dark" ? "text-white hover:text-gray-400" : "text-black hover:text-gray-600"}`}
        onClick={onClose}
      >
        <FaTimes /> {/* ✅ ใช้งานได้แน่นอนแล้ว! */}
      </button>

      {/* ✅ โลโก้ + ปุ่ม Dark/Light Mode */}
      <div className="mt-6 flex items-center justify-between mb-3">
        <span className="text-lg font-bold cursor-pointer hover:text-gray-400 transition">
          Supermhee
        </span>
        <button className="cursor-pointer transition-transform transform hover:scale-110" onClick={toggleTheme}>
          {theme === "dark" ? <FaSun className="text-yellow-400 text-xl" /> : <FaMoon className="text-blue-400 text-xl" />}
        </button>
      </div>

      {/* ✅ ปุ่ม Back ไปเมนูหลัก */}
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
            onClick={() => { navigate("/courses"); onClose(); }} 
            className="w-full flex justify-between items-center text-left hover:text-gray-300 transition"
          >
            📚 Courses <FaChevronRight />
          </button>
        </li>
        <li>
          <button 
            onClick={() => { navigate("/about"); onClose(); }} 
            className="w-full flex justify-between items-center text-left hover:text-gray-300 transition"
          >
            ℹ️ Contact us <FaChevronRight />
          </button>
        </li>
      </ul>
    </div>
  );
};

export default MainMobileMenu;
