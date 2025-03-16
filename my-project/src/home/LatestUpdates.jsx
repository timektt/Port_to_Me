import React from "react";
import { useNavigate } from "react-router-dom";

const latestUpdates = [
  { category: "python-series", level: "101: Basic Python", title: "ทำความรู้จักกับ Python", path: "/courses/python-series/intro", date: "08/10/2567" },
  { category: "python-series", level: "101: Basic Python", title: "Variable & Data Type", path: "/courses/python-series/variables", date: "08/10/2567" },
  { category: "python-series", level: "101: Basic Python", title: "Control Structure", path: "/courses/python-series/control-structure", date: "08/10/2567" },
  { category: "python-series", level: "101: Basic Python", title: "การรับ input & function", path: "/courses/python-series/input-function", date: "08/10/2567" },
  { category: "python-series", level: "101: Basic Python", title: "LeetCode Challenge", path: "/courses/python-series/leetcode", date: "08/10/2567" },
];

const LatestUpdates = ({ theme }) => {
  const navigate = useNavigate();

  return (
    <div className={`latest-updates p-4 sm:p-8 max-w-screen-lg mx-auto w-full`}>
      {/* ✅ Header */}
      <h2 className={`text-xl sm:text-2xl md:text-3xl font-bold text-left mt-6 sm:mt-12 mb-4 sm:mb-6 ${theme === "dark" ? "text-white" : "text-black"}`}>
        📄 Latest Update Documents
      </h2>

      {/* ✅ Container */}
      <div className="space-y-4 w-full">
        {latestUpdates.map((update, index) => (
          <div
            key={index}
            className={`p-4 rounded-lg shadow-lg flex items-center justify-between w-full ${
              theme === "dark" ? "bg-gray-700" : "bg-gray-200"
            } sm:flex-row flex-col`}
          >
            {/* ✅ Tags & Level */}
            <div className="flex items-center space-x-2">
              <span className="bg-green-600 text-white px-2 py-1 rounded text-sm cursor-pointer hover:bg-green-700">
                {update.category}
              </span>
              <span className={`px-2 py-1 rounded text-sm ${theme === "dark" ? "bg-gray-600 text-white" : "bg-gray-400 text-black"}`}>
                {update.level}
              </span>
            </div>

            {/* ✅ Title + อ่าน (คลิกแล้วไปหน้าหัวข้อ Python) */}
            <div className="flex-1 text-left ml-4">
              <h3 className={`font-semibold text-md break-words ${theme === "dark" ? "text-white" : "text-black"}`}>
                {update.title}{" "}
                <button 
                  onClick={() => navigate(update.path)}
                  className="text-green-400 hover:underline hover:text-green-500"
                >
                  อ่าน
                </button>
              </h3>
            </div>

            {/* ✅ Date (อยู่ชิดขวาสุด) */}
            <div className="text-sm sm:text-right text-left sm:w-auto w-full sm:mt-0 mt-2">
              <span className={`${theme === "dark" ? "text-gray-400" : "text-gray-700"}`}>{update.date}</span>
            </div>
          </div>
        ))}
      </div>

      {/* 🔹 เส้นแบ่ง 🔹 */}
      <hr className="my-6 border-t-4 border-gray-300 dark:border-gray-600" />
    </div>
  );
};

export default LatestUpdates;
