import React from "react";

const RestNodejs = () => {
  return (
    <div className="max-w-3xl mx-auto p-6 text-left">
      <h1 className="text-3xl font-bold mb-4">การสร้าง REST API ด้วย Node.js</h1>

      <p className="mb-4">
        Node.js เป็นแพลตฟอร์มที่เหมาะสำหรับการสร้าง RESTful API เนื่องจากรองรับ <strong>JavaScript</strong> และมีไลบรารี 
        เช่น <strong>Express.js</strong> ที่ช่วยให้การพัฒนา API ง่ายขึ้น
      </p>

      <h2 className="text-2xl font-semibold mt-6 mb-2">📌 ติดตั้ง Express.js</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto">
        <code>npm install express</code>
      </pre>

      <h2 className="text-2xl font-semibold mt-6 mb-2">📌 ตัวอย่างโค้ด API เบื้องต้น</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto">
        <code>{`const express = require("express");
const app = express();
app.use(express.json()); // ใช้สำหรับรับ JSON จาก client

// ตัวอย่างฐานข้อมูลจำลอง
const users = [
  { id: 1, name: "John Doe" },
  { id: 2, name: "Jane Doe" }
];

// GET: ดึงข้อมูลผู้ใช้ทั้งหมด
app.get("/api/users", (req, res) => {
  res.json(users);
});

// POST: เพิ่มผู้ใช้ใหม่
app.post("/api/users", (req, res) => {
  const { name } = req.body;
  const newUser = { id: users.length + 1, name };
  users.push(newUser);
  res.status(201).json(newUser);
});

app.listen(3000, () => console.log("Server running on port 3000"));`}</code>
      </pre>

      <p className="mt-4">
        ตัวอย่างนี้แสดงการใช้งาน <code>GET</code> และ <code>POST</code> เพื่อเข้าถึงและเพิ่มข้อมูลผ่าน REST API
      </p>

      <p className="mt-2 text-gray-600 dark:text-gray-300">
        ✅ คุณสามารถทดสอบ API เหล่านี้ได้โดยใช้ Postman หรือ cURL
      </p>
    </div>
  );
};

export default RestNodejs;
