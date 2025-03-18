import React from "react";

const RestNodejs = () => {
  return (
    <div className="max-w-3xl mx-auto p-6 text-left">
      <h1 className="text-3xl font-bold mb-4">การสร้าง REST API ด้วย Node.js</h1>
      <p className="mb-4">
        Node.js เป็นแพลตฟอร์มที่เหมาะสำหรับการสร้าง RESTful API เนื่องจากรองรับ **JavaScript** และมีไลบรารี 
        เช่น Express.js ที่ช่วยให้การพัฒนา API ง่ายขึ้น
      </p>

      <h2 className="text-2xl font-semibold mt-6 mb-2">📌 ติดตั้ง Express.js</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto">
        <code>npm install express</code>
      </pre>

      <h2 className="text-2xl font-semibold mt-6 mb-2">📌 ตัวอย่างโค้ดสร้าง REST API</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto">
        <code>
{`const express = require("express");
const app = express();

app.get("/api/users", (req, res) => {
  res.json([{ id: 1, name: "John Doe" }, { id: 2, name: "Jane Doe" }]);
});

app.listen(3000, () => console.log("Server running on port 3000"));`}
        </code>
      </pre>

      <p className="mt-4">
        API นี้สร้าง **Endpoint** ที่ `GET /api/users` ซึ่งคืนค่ารายการผู้ใช้ในรูปแบบ JSON
      </p>
    </div>
  );
};

export default RestNodejs;
