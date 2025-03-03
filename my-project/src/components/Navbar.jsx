import React, { useState } from "react";
import ProfileInfo from "./ProfileInfo";
import { FaYoutube, FaFacebook, FaGithub, FaBars, FaTimes, FaMoon } from "react-icons/fa";

const Navbar = () => {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <nav className="bg-gray-800 text-white px-4 py-2 flex justify-between items-center text-base md:text-lg">
      {/* Left: Hamburger Menu + Profile */}
      <div className="flex items-center gap-3">
        {/* Hamburger Button */}
        <button className="md:hidden text-white text-3xl" onClick={() => setMenuOpen(true)}>
          <FaBars />
        </button>
        {/* Profile Info */}
        <ProfileInfo />
      </div>

      {/* Right: Social Links + Search Box (เฉพาะจอใหญ่) */}
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
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
