import React from "react";
import { useNavigate } from "react-router-dom";

const tags = [
  { name: "React", count: 4 },
  { name: "Node.js", count: 10 },
  { name: "Data Structure", count: 10 },
  { name: "Algorithm", count: 12 },
  { name: "Backend", count: 14 },
  { name: "SQL", count: 12 },
  { name: "React Component", count: 11 },
  { name: "React Multi Component", count: 26 },
  { name: "Docker", count: 10 },
  { name: "Python", count: 10 },
  { name: "CSS", count: 8 },
  { name: "Javascript", count: 7 },
];

const PopularTags = () => {
  const navigate = useNavigate();

  return (
    <div className="popular-tags p-8 max-w-screen-lg mx-auto w-full">
      <h2 className="text-2xl md:text-3xl font-bold text-white text-left mt-12 mb-6">🏷️ Popular Tags</h2>

      {/* ✅ แสดง Tags */}
      <div className="flex flex-wrap gap-3">
        {tags.map((tag, index) => (
          <div key={index} className="flex items-center bg-gray-800 text-white px-3 py-2 rounded-lg shadow-lg">
            <span className="mr-2">● {tag.name}</span>
            <span className="bg-white text-black px-2 py-1 rounded text-sm">{tag.count}</span>
          </div>
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
