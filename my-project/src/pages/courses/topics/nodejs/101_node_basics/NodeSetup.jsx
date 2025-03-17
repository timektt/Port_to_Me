import React from "react";

const NodeSetup = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-screen-lg mx-auto">
      <h1 className="text-2xl sm:text-3xl font-bold">ติดตั้ง Node.js</h1>
      <p className="mt-4">
        เราสามารถติดตั้ง Node.js ได้จากเว็บไซต์ทางการ:
        <a href="https://nodejs.org" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline ml-1">
          nodejs.org
        </a>
      </p>

      <h2 className="text-xl font-semibold mt-6">🔹 ตรวจสอบเวอร์ชัน Node.js</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        {`node -v`}
      </pre>
    </div>
  );
};

export default NodeSetup;
