import React from "react";

const latestUpdates = [
  { category: "python-series", level: "201: Data", title: "ทำความรู้จักกับ Data", date: "08/10/2567" },
  { category: "python-series", level: "201: Data", title: "ทำความรู้จักกับ Numpy", date: "08/10/2567" },
  { category: "python-series", level: "201: Data", title: "การอ่านไฟล์", date: "08/10/2567" },
  { category: "python-series", level: "201: Data", title: "ทำความรู้จักกับ Pandas", date: "08/10/2567" },
  { category: "python-series", level: "201: Data", title: "Jupyter", date: "08/10/2567" },
];

const LatestUpdates = ({ theme }) => {
  return (
    <div className={`latest-updates p-4 sm:p-8 max-w-screen-lg w-full`}>
      {/* ✅ Header */}
      <h2 className={`text-xl sm:text-2xl md:text-3xl font-bold text-left mt-6 sm:mt-12 mb-4 sm:mb-6 ${theme === "dark" ? "text-white" : "text-black"}`}>
        📄 Latest Update Documents
      </h2>

      {/* ✅ Container */}
      <div className="space-y-4 w-full">
        {latestUpdates.map((update, index) => (
          <div
            key={index}
            className={`p-4 rounded-lg shadow-lg flex flex-col w-full ${
              theme === "dark" ? "bg-gray-700" : "bg-gray-200"
            }`}
          >
            {/* ✅ Top Row (Tags & Date) */}
            <div className="flex justify-between w-full">
              {/* Tags & Level (อยู่ซ้าย) */}
              <div className="flex items-center space-x-2">
                <span className="bg-green-600 text-white px-2 py-1 rounded text-sm cursor-pointer hover:bg-green-700">
                  {update.category}
                </span>
                <span className={`px-2 py-1 rounded text-sm ${theme === "dark" ? "bg-gray-600 text-white" : "bg-gray-400 text-black"}`}>
                  {update.level}
                </span>
              </div>

              {/* Date (อยู่ขวาสุด) */}
              <span className={`text-sm ${theme === "dark" ? "text-gray-400" : "text-gray-700"}`}>{update.date}</span>
            </div>

            {/* ✅ Title + อ่าน (แยกบรรทัด และชิดซ้าย) */}
            <div className="text-left w-full mt-2">
              <h3 className={`font-semibold text-md break-words ${theme === "dark" ? "text-white" : "text-black"}`}>
                {update.title}{" "}
                <a href="#" className="text-green-400 hover:underline hover:text-green-500">อ่าน</a>
              </h3>
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
