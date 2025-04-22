import React, { useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { keywords } from "../../data/keywords"; // ✅ ปรับ path ตามจริง

const generateTagUrl = (tagName) => tagName.toLowerCase().replace(/\s+/g, "-");

// ✅ รายชื่อแท็กหลักที่ต้องการให้แสดงเท่านั้น
const allowedTags = [
  "python",
  "Node.js",
  "graphql",
  "react",
  "Web Development",
  "Basic Programming",
  "Ai",
];

const PopularTags = () => {
  const navigate = useNavigate();

  const tagsWithCount = useMemo(() => {
    const tagMap = {};

    keywords.forEach((item) => {
      if (Array.isArray(item.tags)) {
        item.tags.forEach((tag) => {
          // ✅ นับเฉพาะ tag ที่อยู่ใน allowedTags เท่านั้น
          if (allowedTags.includes(tag)) {
            tagMap[tag] = (tagMap[tag] || 0) + 1;
          }
        });
      }
    });

    return Object.entries(tagMap)
      .map(([name, count]) => ({ name, count }))
      .sort((a, b) => b.count - a.count); // เรียงจากมากไปน้อย
  }, []);

  return (
    <div className="popular-tags p-4 sm:p-6 md:p-8 max-w-screen-lg mx-auto w-full">
      <h2 className="text-xl sm:text-2xl md:text-3xl font-bold text-white text-left mt-10 mb-6">
        Tags
      </h2>

      <div className="flex flex-wrap gap-3 justify-start sm:justify-start">
        {tagsWithCount.map((tag) => (
          <button
            key={tag.name}
            onClick={() => navigate(`/tags/${generateTagUrl(tag.name)}`)}
            className="flex justify-between items-center min-w-[140px] w-full sm:w-auto text-sm sm:text-base bg-gray-800 text-white px-3 py-2 rounded-lg shadow-md hover:bg-gray-700 transition"
            aria-label={`Navigate to ${tag.name} tag`}
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
        aria-label="View all tags"
      >
        View all Tags
      </button>
    </div>
  );
};

export default PopularTags;
