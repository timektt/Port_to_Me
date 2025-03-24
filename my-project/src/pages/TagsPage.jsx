import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

// ✅ เปลี่ยนเป็น object: { name, query }
const allTags = [
  { name: "Python", query: "python" },
  { name: "Node.js", query: "nodejs" },
  { name: "GraphQL", query: "graphql" },
  { name: "React", query: "react" },
  { name: "Web Development", query: "web" },
  { name: "Basic Programming", query: "basic" },
];

const TagsPage = () => {
  const [theme, setTheme] = useState(document.documentElement.classList.contains("dark") ? "dark" : "light");
  const navigate = useNavigate();

  useEffect(() => {
    const observer = new MutationObserver(() => {
      setTheme(document.documentElement.classList.contains("dark") ? "dark" : "light");
    });

    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => observer.disconnect();
  }, []);

  return (
    <div className={`tags-page min-h-screen p-4 sm:p-6 md:p-8 max-w-screen-lg mx-auto w-full relative z-10 mt-[5rem] 
      ${theme === "dark" ? "bg-gray-900 text-white" : "bg-white text-gray-900"}`}>

      <h1 className="text-xl sm:text-2xl md:text-3xl lg:text-4xl font-bold mb-6 text-center sm:text-left">
        🏷️ All Tags
      </h1>

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
            {allTags.map((tag, index) => (
        <button
          key={index}
          onClick={() => navigate(`/tags/${tag.query}`)}
          className={`px-4 py-2 rounded-lg shadow-md cursor-pointer text-center transition-all duration-300
            ${theme === "dark" ? "bg-gray-800 text-white hover:bg-gray-700" : "bg-gray-200 text-black hover:bg-gray-300"}`}
        >
          {tag.name}
        </button>
      ))}

      </div>
    </div>
  );
};

export default TagsPage;
