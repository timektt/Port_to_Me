import React from "react";

const latestUpdates = [
  { category: "python-series", level: "201: Data", title: "ทำความรู้จักกับ Data", date: "08/10/2567" },
  { category: "python-series", level: "201: Data", title: "ทำความรู้จักกับ Numpy", date: "08/10/2567" },
  { category: "python-series", level: "201: Data", title: "การอ่านไฟล์", date: "08/10/2567" },
  { category: "python-series", level: "201: Data", title: "ทำความรู้จักกับ Pandas", date: "08/10/2567" },
  { category: "python-series", level: "201: Data", title: "Jupyter", date: "08/10/2567" },
  { category: "python-series", level: "101: Basic Python", title: "แนะนำ Python", date: "26/08/2567" },
];

const LatestUpdates = () => {
  return (
    <div className="latest-updates p-8 max-w-screen-lg mx-auto w-full">
      <h2 className="text-2xl md:text-3xl font-bold text-white text-left mt-12 mb-6">
        📄 Latest Update Documents
      </h2>
      <div className="space-y-4 w-full">
        {latestUpdates.map((update, index) => (
          <div key={index} className="bg-gray-800 p-4 rounded-lg shadow-lg flex justify-between items-center transition-transform transform hover:scale-105 w-full">
            <div>
              <span className="bg-green-600 text-white px-2 py-1 rounded text-sm mr-2">{update.category}</span>
              <span className="bg-gray-600 text-white px-2 py-1 rounded text-sm">{update.level}</span>
              <h3 className="text-white font-semibold text-md mt-2">
                {update.title} <a href="#" className="text-green-400">อ่าน</a>
              </h3>
            </div>
            <span className="text-gray-400 text-sm">{update.date}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default LatestUpdates;
