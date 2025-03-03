import React from "react";

const tags = [
  { name: "vue", count: 78 },
  { name: "c++", count: 48 },
  { name: "data structure", count: 48 },
  { name: "algorithm", count: 48 },
  { name: "backend", count: 43 },
  { name: "go", count: 41 },
  { name: "firebase", count: 38 },
  { name: "vue component", count: 31 },
  { name: "vue multi component", count: 26 },
  { name: "cloud firestore", count: 15 },
  { name: "cloud function", count: 14 },
  { name: "docker", count: 10 },
  { name: "python", count: 10 },
  { name: "css", count: 8 },
  { name: "javascript", count: 7 },
];

const PopularTags = () => {
  return (
    <div className="popular-tags p-8 max-w-screen-lg mx-auto w-full">
      <h2 className="text-2xl md:text-3xl font-bold text-white text-left mt-12 mb-6">🏷️ Popular Tags</h2>
      <div className="flex flex-wrap gap-3">
        {tags.map((tag, index) => (
          <div
            key={index}
            className="flex items-center bg-gray-800 text-white px-3 py-2 rounded-lg shadow-lg transition-transform transform hover:scale-105"
          >
            <span className="mr-2">● {tag.name}</span>
            <span className="bg-white text-black px-2 py-1 rounded text-sm">{tag.count}</span>
          </div>
        ))}
      </div>
      <a href="#" className="text-green-400 text-sm md:text-base mt-4 inline-block hover:underline">
        View all tags
      </a>
    </div>
  );
};

export default PopularTags;
