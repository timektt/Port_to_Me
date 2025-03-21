import React from "react";

const RestApiBasics = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold">🌐 REST API Basics</h1>
      <p className="mt-4">
        REST (Representational State Transfer) เป็นแนวทางในการออกแบบ Web API ที่เน้นความเรียบง่าย ใช้ <strong>HTTP Methods</strong> เป็นหลัก:
      </p>
      <ul className="list-disc ml-5 mt-2">
        <li><strong>GET:</strong> ดึงข้อมูล</li>
        <li><strong>POST:</strong> สร้างข้อมูล</li>
        <li><strong>PUT:</strong> แก้ไขข้อมูล</li>
        <li><strong>DELETE:</strong> ลบข้อมูล</li>
      </ul>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่างการสร้าง REST API ด้วย Express</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto text-sm">
        <code>{`const express = require("express");
const app = express();
app.use(express.json());

// Fake database
const users = [
  { id: 1, name: "John Doe" },
  { id: 2, name: "Jane Smith" }
];

// GET - ดึงผู้ใช้ทั้งหมด
app.get("/api/users", (req, res) => {
  res.json(users);
});

// POST - เพิ่มผู้ใช้ใหม่
app.post("/api/users", (req, res) => {
  const newUser = { id: users.length + 1, ...req.body };
  users.push(newUser);
  res.status(201).json(newUser);
});

// PUT - แก้ไขข้อมูลผู้ใช้
app.put("/api/users/:id", (req, res) => {
  const id = parseInt(req.params.id);
  const index = users.findIndex(user => user.id === id);
  if (index !== -1) {
    users[index] = { id, ...req.body };
    res.json(users[index]);
  } else {
    res.status(404).json({ message: "User not found" });
  }
});

// DELETE - ลบผู้ใช้
app.delete("/api/users/:id", (req, res) => {
  const id = parseInt(req.params.id);
  const index = users.findIndex(user => user.id === id);
  if (index !== -1) {
    const deleted = users.splice(index, 1);
    res.json(deleted[0]);
  } else {
    res.status(404).json({ message: "User not found" });
  }
});

app.listen(3000, () => console.log("🚀 Server running on http://localhost:3000"));`}</code>
      </pre>

      <p className="mt-6 text-gray-700 dark:text-gray-300">
        🔍 คุณสามารถใช้ <strong>Postman</strong>, <strong>Insomnia</strong> หรือ <code>curl</code> เพื่อทดสอบแต่ละ Endpoint ได้
      </p>
    </div>
  );
};

export default RestApiBasics;
