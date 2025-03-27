import React, { useState, useEffect } from "react";
import { useNavigate, useLocation, Link } from "react-router-dom";
import ProfileInfo from "../../profile/ProfileInfo";
import {
  FaYoutube,
  FaFacebook,
  FaGithub,
  FaBars,
  FaMoon,
  FaSearch,
} from "react-icons/fa";
import { IoSunny } from "react-icons/io5";
import MainMobileMenu from "../../menu/MainMobileMenu";
import PythonMobileMenu from "./sidebar/MobileMenus/PythonMobileMenu";
import BasicProgrammingMobileMenu from "./sidebar/MobileMenus/BasicProgrammingMobileMenu";
import NodeMobileMenu from "./sidebar/MobileMenus/NodeMobileMenu";
import RestfulApiGraphQLMobileMenu from "./sidebar/MobileMenus/RestfulApiGraphQLMobileMenu";
import ReactJsMobileMenu from "./sidebar/MobileMenus/ReactJsMobileMenu";
import WebDevMobileMenu from "./sidebar/MobileMenus/WebDevMobileMenu";

const Navbar = ({ theme, setTheme }) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const navigate = useNavigate();
  const location = useLocation();

  const isLoggedIn = !!localStorage.getItem("token");
  const handleLogout = () => {
    localStorage.removeItem("token");
    navigate("/");
  };

  useEffect(() => {
    setSearchQuery("");
  }, [location.pathname]);

  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth >= 988) {
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
    <nav
      className={`navbar px-4 py-2 h-16 flex justify-between items-center text-base md:text-lg fixed top-0 w-full z-50 ${
        theme === "dark" ? "bg-gray-800 text-white" : "bg-white text-gray-900"
      }`}
    >
      {/* ✅ Left Section: Hamburger Menu + Profile */}
      <div className="flex items-center gap-3">
        <button className="md:hidden text-3xl" onClick={() => setMobileMenuOpen(true)}>
          <FaBars />
        </button>
        <ProfileInfo navigate={navigate} />
      </div>

      {/* ✅ Mobile Search Input */}
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

        <div className="relative w-48">
          <input
            type="text"
            placeholder="Search..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && searchQuery.trim()) {
                navigate(`/search?q=${encodeURIComponent(searchQuery.trim())}`);
              }
            }}
            className={`p-2 pl-8 w-full rounded-lg border focus:outline-none focus:ring-2 focus:ring-blue-500 ${
              theme === "dark"
                ? "bg-gray-700 text-white border-gray-600"
                : "bg-gray-200 text-gray-900 border-gray-400"
            }`}
          />
          <FaSearch className="absolute left-2 top-3 text-gray-400" />
        </div>

{/* ✅ Login/Logout ปุ่มดีไซน์ใหม่ */}
{isLoggedIn ? (
  <button
    onClick={handleLogout}
    className="px-5 py-2 rounded-full font-semibold text-white bg-red-500 hover:bg-red-600 shadow-md transition duration-300"
  >
    Logout
  </button>
) : (
  <Link
    to="/login"
    className="px-5 py-2 rounded-full font-semibold text-white bg-gray-500 hover:bg-gray-600 shadow-md transition duration-300"
  >
    Login
  </Link>
        )}
      </div>

      {/* ✅ Mobile Menu Overlay */}
      {mobileMenuOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50">
          {location.pathname.startsWith("/courses/python-series") ? (
            <PythonMobileMenu onClose={() => setMobileMenuOpen(false)} theme={theme} setTheme={setTheme} />
          ) : location.pathname.startsWith("/courses/nodejs-series") ? (
            <NodeMobileMenu onClose={() => setMobileMenuOpen(false)} theme={theme} setTheme={setTheme} />
          ) : location.pathname.startsWith("/courses/restful-api-graphql-series") ? (
            <RestfulApiGraphQLMobileMenu onClose={() => setMobileMenuOpen(false)} theme={theme} setTheme={setTheme} />
          ) : location.pathname.startsWith("/courses/reactjs-series") ? (
            <ReactJsMobileMenu onClose={() => setMobileMenuOpen(false)} theme={theme} setTheme={setTheme} />
          ) : location.pathname.startsWith("/courses/web-development") ? (
            <WebDevMobileMenu onClose={() => setMobileMenuOpen(false)} theme={theme} setTheme={setTheme} />
          ) : location.pathname.startsWith("/courses/basic-programming") ? (
            <BasicProgrammingMobileMenu onClose={() => setMobileMenuOpen(false)} theme={theme} setTheme={setTheme} />
          ) : (
            <MainMobileMenu onClose={() => setMobileMenuOpen(false)} theme={theme} setTheme={setTheme} />
          )}
        </div>
      )}
    </nav>
  );
};

export default Navbar;
