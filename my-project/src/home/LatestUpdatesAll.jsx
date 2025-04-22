import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

const LatestUpdatesAll = ({ theme }) => {
  const [allUpdates, setAllUpdates] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchUpdates = async () => {
      try {
        const res = await fetch("/data/updates.json");
        const data = await res.json();
        setAllUpdates(data);
      } catch (err) {
        console.error("❌ Error loading updates.json:", err);
      }
    };

    fetchUpdates();
  }, []);

  const sortedUpdates = allUpdates.sort((a, b) => new Date(b.date) - new Date(a.date));

  return (
    <div className="all-updates p-4 sm:p-8 max-w-screen-lg mx-auto w-full">
      <h2
        className={`text-xl sm:text-2xl md:text-3xl font-bold text-left mt-6 sm:mt-12 mb-4 sm:mb-6 ${
          theme === "dark" ? "text-white" : "text-black"
        }`}
      >
        เอกสารอัปเดตทั้งหมด
      </h2>

      <div className="space-y-4 w-full">
        {sortedUpdates.map((update, index) => (
          <div
            key={index}
            className={`p-4 rounded-lg shadow-lg flex flex-col sm:flex-row items-start sm:items-center w-full ${
              theme === "dark" ? "bg-gray-800" : "bg-gray-200"
            }`}
          >
            <div className="flex flex-wrap sm:flex-nowrap items-center space-x-2">
              <span className="bg-green-600 text-white px-2 py-1 rounded text-sm cursor-pointer hover:bg-green-700">
                {update.category}
              </span>
              <span
                className={`px-2 py-1 rounded text-sm ${
                  theme === "dark" ? "bg-gray-600 text-white" : "bg-gray-400 text-black"
                }`}
              >
                {update.level}
              </span>
            </div>

            <div className="w-auto sm:w-auto text-left mt-2 sm:mt-0">
              <h3
                className={`font-semibold text-md break-words ${
                  theme === "dark" ? "text-white" : "text-black"
                }`}
              >
                {update.title}{" "}
                <button
                  onClick={() => navigate(update.path)}
                  className="text-green-400 hover:underline hover:text-green-500"
                >
                  อ่าน
                </button>
              </h3>
            </div>

            <div className="text-sm text-left sm:text-right sm:w-auto w-full sm:ml-auto mt-2 sm:mt-0">
              <span className={theme === "dark" ? "text-gray-400" : "text-gray-700"}>
                {update.date}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default LatestUpdatesAll;
