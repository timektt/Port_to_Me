import React, { useState } from "react";
import ProfileInfo from "./ProfileInfo";
import { FaYoutube, FaFacebook, FaGithub, FaBars, FaTimes, FaMoon } from "react-icons/fa";

const Navbar = () => {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <nav className="bg-gray-800 text-white px-4 py-2 flex justify-between items-center">
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
          <FaYoutube className="text-2xl hover:text-red-500" />
        </a>
        <a href="https://facebook.com" target="_blank" rel="noopener noreferrer">
          <FaFacebook className="text-2xl hover:text-blue-500" />
        </a>
        <a href="https://github.com" target="_blank" rel="noopener noreferrer">
          <FaGithub className="text-2xl hover:text-gray-500" />
        </a>
        <div className="relative">
          <input
            type="text"
            placeholder="Search..."
            className="p-2 pl-8 rounded-lg bg-gray-700 text-white border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
      </div>

      {/* Fullscreen Sidebar Menu */}
      {menuOpen && (
        <div className="fixed inset-0 bg-gray-900 text-white flex flex-col p-6 w-64 shadow-lg">
          {/* Close Button */}
          <button className="self-end text-3xl" onClick={() => setMenuOpen(false)}>
            <FaTimes />
          </button>

          {/* Profile */}
          <div className="flex items-center gap-4 mt-4">
            <img src="/spm.png" alt="Profile" className="w-12 h-12 rounded-full" />
            <h1 className="text-lg font-bold">Supermhee</h1>
            <FaMoon className="ml-auto text-2xl cursor-pointer" />
          </div>

          {/* Menu Links */}
          <div className="mt-6 flex flex-col gap-6 text-lg">
            <a href="#" className="hover:text-gray-400 flex items-center gap-3">
              Courses
            </a>
            <a href="#" className="hover:text-gray-400 flex items-center gap-3">
              Posts
            </a>
            <a href="https://youtube.com" target="_blank" rel="noopener noreferrer" className="hover:text-gray-400 flex items-center gap-3">
              <FaYoutube className="text-3xl" /> Youtube
            </a>
            <a href="https://facebook.com" target="_blank" rel="noopener noreferrer" className="hover:text-gray-400 flex items-center gap-3">
              <FaFacebook className="text-3xl" /> Facebook
            </a>
            <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="hover:text-gray-400 flex items-center gap-3">
              <FaGithub className="text-3xl" /> Github
            </a>
          </div>
        </div>
      )}
    </nav>
  );
};

export default Navbar;
