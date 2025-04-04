import React from "react";
import {
  FaTimes,
  FaChevronRight,
  FaMoon,
  FaHome,
  FaTags,
  FaPhoneAlt,
  FaSignOutAlt,
  FaUserCircle,
} from "react-icons/fa";
import { FiSun } from "react-icons/fi";
import { useNavigate } from "react-router-dom";

const MainMobileMenu = ({ onClose, theme, setTheme }) => {
  const navigate = useNavigate();
  const isLoggedIn = !!localStorage.getItem("token");

  const toggleTheme = () => {
    const newTheme = theme === "dark" ? "light" : "dark";
    setTheme(newTheme);
    localStorage.setItem("theme", newTheme);
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    navigate("/");
    onClose();
  };

  return (
    <>
      {/* ✅ Overlay */}
      <div
        className="fixed inset-0 bg-black bg-opacity-30 z-40"
        onClick={onClose}
        aria-label="Close menu overlay"
      ></div>

      {/* ✅ Main Menu with scroll */}
      <div
        className={`fixed top-0 left-0 w-64 h-full z-50 shadow-lg transition-transform duration-300 ${
          theme === "dark" ? "bg-gray-900 text-white" : "bg-white text-black"
        }`}
      >
        <div className="h-full overflow-y-auto p-4">
          {/* ✅ Close Button */}
          <button
            className={`absolute right-4 top-4 text-2xl transition-colors duration-200 ${
              theme === "dark"
                ? "text-white hover:text-gray-400"
                : "text-black hover:text-gray-600"
            }`}
            onClick={onClose}
          >
            <FaTimes />
          </button>

          {/* ✅ Logo + Theme Toggle */}
          <div className="mt-6 flex items-center mb-6">
            <img
              src="/spm2.jpg"
              alt="Logo"
              className="w-10 h-10 mr-3 object-cover rounded-full"
            />
            <div className="flex items-center space-x-2">
              <span className="text-lg font-bold cursor-pointer hover:text-gray-400 transition">
                Superbear
              </span>
              <button
                onClick={toggleTheme}
                className="cursor-pointer transition-transform transform hover:scale-110"
                aria-label="Toggle dark mode"
              >
                {theme === "dark" ? (
                  <FiSun className="text-yellow-400 text-2xl" />
                ) : (
                  <FaMoon className="text-blue-400 text-2xl" />
                )}
              </button>
            </div>
          </div>

          {/* ✅ Menu List */}
          <ul className="mt-3 space-y-4">
            <li>
              <button
                onClick={() => {
                  navigate("/courses");
                  onClose();
                }}
                className="w-full flex justify-between items-center text-left hover:text-gray-300 transition"
              >
                <span className="flex items-center gap-2">
                  <FaHome className="text-lg" /> Courses
                </span>
                <FaChevronRight />
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
                <span className="flex items-center gap-2">
                  <FaTags className="text-lg" /> Tags
                </span>
                <FaChevronRight />
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
                <span className="flex items-center gap-2">
                  <FaPhoneAlt className="text-lg" /> Contact Us
                </span>
                <FaChevronRight />
              </button>
            </li>
          </ul>

          {/* ✅ User Section */}
          <div className="mt-8 space-y-4">
            {isLoggedIn ? (
              <>
                <button
                  onClick={() => {
                    navigate("/dashboard");
                    onClose();
                  }}
                  className="w-full py-2 rounded-full font-semibold text-yellow-400 bg-black 
                    border border-yellow-500 shadow-md hover:shadow-yellow-300/50 transition duration-300 flex items-center justify-center gap-2"
                >
                  <FaUserCircle className="text-lg" /> Dashboard
                </button>

                <button
                  onClick={() => {
                    navigate("/profile");
                    onClose();
                  }}
                  className="w-full py-2 bg-gradient-to-r from-indigo-500 to-blue-500 text-white rounded-full shadow-md hover:from-indigo-600 hover:to-blue-600 transition duration-300 flex items-center justify-center gap-2"
                >
                  <FaUserCircle className="text-lg" /> Profile
                </button>

                <button
                  onClick={handleLogout}
                  className="w-full py-2 bg-red-500 text-white rounded-full hover:bg-red-600 transition flex items-center justify-center gap-2"
                >
                  <FaSignOutAlt className="text-lg" /> Logout
                </button>
              </>
            ) : (
              <button
                onClick={() => {
                  navigate("/login");
                  onClose();
                }}
                className="w-full py-2 bg-gray-500 text-white rounded-full hover:bg-gray-600 transition flex items-center justify-center gap-2"
              >
                <FaUserCircle className="text-lg" /> Login
              </button>
            )}
          </div>
        </div>
      </div>
    </>
  );
};

export default MainMobileMenu;
