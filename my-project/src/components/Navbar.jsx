import React from "react";
import { FaYoutube, FaFacebook, FaGithub, FaSearch } from "react-icons/fa";

const Navbar = () => {
  return (
    <nav className="bg-gray-800 text-white p-4 flex justify-between items-center">
      {/* Left Side: Profile Info */}
      <div className="flex items-center gap-3">
        <img
          src="/profile.jpg"
          alt="Profile"
          className="w-12 h-12 rounded-full"
        />
        <div>
          <h1 className="text-lg font-bold">Your Name</h1>
          <p className="text-sm">Your Course | Your Post</p>
        </div>
      </div>
      
      {/* Right Side: Social Links & Search Box */}
      <div className="flex items-center gap-4">
        <a href="https://youtube.com" target="_blank" rel="noopener noreferrer">
          <FaYoutube className="text-xl hover:text-red-500" />
        </a>
        <a href="https://facebook.com" target="_blank" rel="noopener noreferrer">
          <FaFacebook className="text-xl hover:text-blue-500" />
        </a>
        <a href="https://github.com" target="_blank" rel="noopener noreferrer">
          <FaGithub className="text-xl hover:text-gray-500" />
        </a>

        {/* Search Box */}
        <div className="relative">
          <input
            type="text"
            placeholder="Search..."
            className="p-2 pl-8 rounded-lg bg-gray-700 text-white border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <FaSearch className="absolute left-2 top-2.5 text-gray-400" />
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
