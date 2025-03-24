import React from "react";
import { useNavigate } from "react-router-dom";

// ✅ ใช้ name ที่แสดงผล และ query สำหรับลิงก์
const tags = [
  { name: "Python", query: "python", count: 21 },
  { name: "Node.js", query: "nodejs", count: 30 },
  { name: "GraphQL", query: "graphql", count: 25 },
  { name: "React", query: "react", count: 30 },
  { name: "Web Development", query: "web", count: 30 },
  { name: "Basic Programming", query: "basic", count: 29 },
];

const PopularTags = () => {
  const navigate = useNavigate();

  return (
    <div className="popular-tags p-8 max-w-screen-lg mx-auto w-full">
      <h2 className="text-2xl md:text-3xl font-bold text-white text-left mt-12 mb-6"> Popular Tags</h2>

      {/* ✅ แสดง Tags พร้อมลิงก์ */}
      <div className="flex flex-wrap gap-3">
        {tags.map((tag, index) => (
          <button
            key={index}
            onClick={() => navigate(`/tags/${tag.name.toLowerCase().replace(/\s+/g, "-")}`)}
            className="flex items-center bg-gray-800 text-white px-3 py-2 rounded-lg shadow-lg hover:bg-gray-700 transition"
          >
            <span className="mr-2">● {tag.name}</span>
            <span className="bg-white text-black px-2 py-1 rounded text-sm">{tag.count}</span>
          </button>
        ))}
      </div>

      {/* ✅ ปุ่ม View all Tags */}
      <button
        onClick={() => navigate("/tags")}
        className="text-green-400 text-sm md:text-base mt-4 inline-block hover:underline"
      >
        View all Tags
      </button>
    </div>
  );
};

export default PopularTags;
