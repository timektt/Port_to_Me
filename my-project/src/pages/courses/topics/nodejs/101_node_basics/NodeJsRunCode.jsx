import React from "react";

const NodeJsRunCode = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-screen-lg mx-auto">
      <h1 className="text-2xl sm:text-3xl font-bold">รัน JavaScript ใน Node.js</h1>
      <p className="mt-4">
        สามารถรันโค้ด JavaScript บน Node.js ได้โดยใช้ Terminal หรือ Command Line
      </p>

      <h2 className="text-xl font-semibold mt-6">🔹 สร้างไฟล์ JavaScript</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        {`// app.js
console.log("Hello from Node.js");`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">🔹 รันโค้ดใน Terminal</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        {`node app.js`}
      </pre>
    </div>
  );
};

export default NodeJsRunCode;
