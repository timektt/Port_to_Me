import React, { useState, useEffect, useContext, useRef } from "react";
import { useNavigate, useLocation, Link } from "react-router-dom";
import ProfileInfo from "../../profile/ProfileInfo";
import {
  FaYoutube,
  FaFacebook,
  FaGithub,
  FaBars,
  FaMoon,
  FaSearch,
  FaUserCircle,
  FaSignOutAlt,
  FaHome,
  FaTimes,
  FaTags,
  FaPhoneAlt,
} from "react-icons/fa";
import { IoSunny } from "react-icons/io5";
import MainMobileMenu from "../../menu/MainMobileMenu";
import PythonMobileMenu from "./sidebar/MobileMenus/PythonMobileMenu";
import BasicProgrammingMobileMenu from "./sidebar/MobileMenus/BasicProgrammingMobileMenu";
import NodeMobileMenu from "./sidebar/MobileMenus/NodeMobileMenu";
import RestfulApiGraphQLMobileMenu from "./sidebar/MobileMenus/RestfulApiGraphQLMobileMenu";
import ReactJsMobileMenu from "./sidebar/MobileMenus/ReactJsMobileMenu";
import WebDevMobileMenu from "./sidebar/MobileMenus/WebDevMobileMenu";
import { AuthContext } from "../context/AuthContext";

const Navbar = ({ theme, setTheme }) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [showDropdown, setShowDropdown] = useState(false);
  const dropdownRef = useRef(null);
  const navigate = useNavigate();
  const location = useLocation();
  const { user, logout } = useContext(AuthContext);

  const handleLogout = async () => {
    await logout();
    window.location.href = "/";
  };

  useEffect(() => {
    setSearchQuery("");
    setShowDropdown(false);
  }, [location.pathname]);

  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth >= 988) {
        setMobileMenuOpen(false);
      }
    };
    const handleClickOutside = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setShowDropdown(false);
      }
    };
    window.addEventListener("resize", handleResize);
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      window.removeEventListener("resize", handleResize);
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  const toggleTheme = () => {
    setTheme(theme === "dark" ? "light" : "dark");
  };

  return (
    <nav
      className={`navbar px-4 py-2 h-16 flex justify-between items-center text-base md:text-lg fixed top-0 w-full z-50 ${
        theme === "dark"
          ? "bg-gradient-to-b from-gray-900 to-gray-600 text-white"
          : "bg-white text-gray-900"
      }`}
    >
      {/* ✅ Left Section */}
      <div className="flex items-center gap-3">
        <button className="md:hidden text-3xl" onClick={() => setMobileMenuOpen(true)}>
          <FaBars />
        </button>
        <ProfileInfo navigate={navigate} />
      </div>

      {/* ✅ Mobile Search */}
      <div className="absolute right-4 top-1/2 transform -translate-y-1/2 w-48 md:hidden">
        <div className="relative">
          <input
            type="text"
            placeholder="Search..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && searchQuery.trim()) {
                navigate(`/search?q=${encodeURIComponent(searchQuery.trim())}`);
                setSearchQuery("");
                setMobileMenuOpen(false);
              }
            }}
            className={`p-2 pl-9 w-full rounded-md border focus:outline-none focus:ring-2 focus:ring-blue-500 transition ${
              theme === "dark"
                ? "bg-gray-700 text-white border-gray-600"
                : "bg-gray-200 text-gray-900 border-gray-400"
            }`}
          />
          <FaSearch className="absolute left-3 top-3 text-gray-400 pointer-events-none" />
        </div>
      </div>

      {/* ✅ Desktop Right Section */}
      <div className="hidden md:flex items-center gap-6">
        <a href="https://youtube.com" target="_blank" rel="noopener noreferrer" className="hover:text-red-500">
          <FaYoutube className="text-2xl" />
        </a>
        <a href="https://facebook.com" target="_blank" rel="noopener noreferrer" className="hover:text-blue-500">
          <FaFacebook className="text-2xl" />
        </a>
        <a href="https://github.com/timektt" target="_blank" rel="noopener noreferrer" className="hover:text-gray-500">
          <FaGithub className="text-2xl" />
        </a>

        <button
          onClick={toggleTheme}
          className="w-10 h-10 flex items-center justify-center rounded-full transition bg-gray-700 hover:bg-gray-600"
        >
          {theme === "dark" ? (
            <IoSunny className="text-yellow-300 text-2xl" />
          ) : (
            <FaMoon className="text-blue-400 text-2xl" />
          )}
        </button>

        {user ? (
          <div className="relative" ref={dropdownRef}>
            {/* ปุ่มเปิด/ปิด Dropdown */}
            <button
              onClick={() => setShowDropdown(!showDropdown)}
              className="flex items-center gap-2"
            >
              <FaUserCircle className="text-3xl text-gray-400" />
              <span className="font-medium hidden lg:block">
                {user.displayName?.split(" ")[0]}
              </span>
            </button>

            {/* Dropdown */}
            {showDropdown && (
              <div
                className={`absolute right-0 top-full mt-1 w-44 bg-white text-black shadow-lg z-50 transition-all duration-300 transform ${
                  showDropdown ? "opacity-100 scale-100" : "opacity-0 scale-95"
                } rounded-b-lg rounded-t-md`}
                style={{
                  borderTopLeftRadius: "0.375rem", // ขอบบนเหลี่ยม
                  borderTopRightRadius: "0.375rem", // ขอบบนเหลี่ยม
                  borderBottomLeftRadius: "0.75rem", // ขอบล่างมล
                  borderBottomRightRadius: "0.75rem", // ขอบล่างมล
                }}
              >
                <Link
                  to="/dashboard"
                  className="block px-4 py-2 hover:bg-gray-100 flex items-center gap-2"
                >
                  <FaHome className="text-gray-500" /> Dashboard
                </Link>
                <Link
                  to="/profile"
                  className="block px-4 py-2 hover:bg-gray-100 flex items-center gap-2"
                >
                  <FaUserCircle className="text-gray-500" /> Profile
                </Link>
                <button
                  onClick={handleLogout}
                  className="block w-full text-left px-4 py-2 hover:bg-gray-100 flex items-center gap-2"
                >
                  <FaSignOutAlt className="text-gray-500" /> Logout
                </button>
              </div>
            )}
          </div>
        ) : (
          <Link
            to="/login"
            className="px-5 py-2 rounded-full font-semibold text-yellow-400 bg-black 
              border border-yellow-500 transition-all duration-300
              hover:shadow-[0_0_6px_#FFD700,0_0_12px_#FFD700]"
          >
            Login
          </Link>
        )}
      </div>

      {mobileMenuOpen && (
  <MainMobileMenu
    onClose={() => setMobileMenuOpen(false)}
    theme={theme}
    setTheme={setTheme}
  />
)}
    </nav>
  );
};

export default Navbar;
