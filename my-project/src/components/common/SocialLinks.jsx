import React from "react";
import { FaYoutube, FaFacebook, FaGithub, FaSearch } from "react-icons/fa";

const SocialLinks = () => {
  return (
    <div className="flex items-center gap-4 ">
      <a href="https://youtube.com" target="_blank" rel="noopener noreferrer">
        <FaYoutube className="text-3xl hover:text-red-500" />
      </a>
      <a href="https://facebook.com" target="_blank" rel="noopener noreferrer">
        <FaFacebook className="text-3xl hover:text-blue-500" />
      </a>
      <a href="https://github.com/timektt" target="_blank" rel="noopener noreferrer">
        <FaGithub className="text-3xl hover:text-gray-500" />
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
  );
};

export default SocialLinks;
