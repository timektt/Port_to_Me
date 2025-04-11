import React from "react";
import { useNavigate } from "react-router-dom";

// ฟังก์ชันช่วยเหลือสำหรับแปลงชื่อแท็กเป็น URL-friendly string
const generateTagUrl = (tagName) => tagName.toLowerCase().replace(/\s+/g, "-");

const tags = [
  { name: "Python", query: "python", count: 21 },
  { name: "Node.js", query: "nodejs", count: 30 },
  { name: "GraphQL", query: "graphql", count: 25 },
  { name: "React", query: "react", count: 30 },
  { name: "Web Development", query: "web", count: 30 },
  { name: "Basic Programming", query: "basic", count: 29 },
  { name: "Ai", query: "Ai", count: 5 },
];

const PopularTags = () => {
  const navigate = useNavigate();

  return (
    <div className="popular-tags p-4 sm:p-6 md:p-8 max-w-screen-lg mx-auto w-full">
      <h2 className="text-xl sm:text-2xl md:text-3xl font-bold text-white text-left mt-10 mb-6">
        Popular Tags
      </h2>

      <div className="flex flex-wrap gap-3 justify-start sm:justify-start">
        {tags.map((tag) => (
          <button
            key={tag.name} // ใช้ tag.name เป็น key เพื่อความไม่ซ้ำ
            onClick={() => navigate(`/tags/${generateTagUrl(tag.name)}`)} // ใช้ฟังก์ชันช่วยเหลือ
            className="flex justify-between items-center min-w-[140px] w-full sm:w-auto text-sm sm:text-base bg-gray-800 text-white px-3 py-2 rounded-lg shadow-md hover:bg-gray-700 transition"
            aria-label={`Navigate to ${tag.name} tag`} // เพิ่ม aria-label
          >
            <span className="mr-2 truncate">● {tag.name}</span>
            <span className="bg-white text-black px-2 py-1 rounded text-xs sm:text-sm">
              {tag.count}
            </span>
          </button>
        ))}
      </div>

      <button
        onClick={() => navigate("/tags")}
        className="text-green-400 text-sm md:text-base mt-5 inline-block hover:underline"
        aria-label="View all tags" // เพิ่ม aria-label
      >
        View all Tags
      </button>
    </div>
  );
};

export default PopularTags;
