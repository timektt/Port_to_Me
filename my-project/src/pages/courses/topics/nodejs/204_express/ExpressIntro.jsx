import React from "react";

const ExpressIntro = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold">🚀 Introduction to Express.js</h1>

      <p className="mt-4 text-gray-700 dark:text-gray-300">
        Express.js เป็น <strong>Node.js Web Framework</strong> ที่เรียบง่ายและมีประสิทธิภาพสูง ใช้สำหรับพัฒนา Web App, RESTful API และ Microservices ได้อย่างรวดเร็ว
      </p>

      <h2 className="text-xl font-semibold mt-6">📦 วิธีติดตั้ง Express.js</h2>
      <p className="mt-2">ติดตั้งผ่านคำสั่งด้านล่างนี้:</p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`npm init -y
npm install express`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">📄 สร้างไฟล์ server.js</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`const express = require("express");
const app = express();

app.get("/", (req, res) => {
  res.send("Hello, Express!");
});

app.listen(3000, () => console.log("🚀 Server running on port 3000"));`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">🧪 ทดสอบ</h2>
      <p className="mt-2">เปิดเบราว์เซอร์หรือใช้ Postman เข้าลิงก์นี้:</p>
      <pre className="bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-white p-3 rounded">
        http://localhost:3000/
      </pre>

      <h2 className="text-xl font-semibold mt-6">🛠️ เพิ่ม Route เพิ่มเติม</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`app.get("/about", (req, res) => {
  res.send("This is About Page");
});

app.post("/contact", (req, res) => {
  res.send("Contact form submitted");
});`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">🧩 การใช้ Middleware เบื้องต้น</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`app.use(express.json()); // เพื่อรองรับ JSON body

app.post("/api", (req, res) => {
  console.log(req.body);
  res.json({ status: "OK", data: req.body });
});`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">📁 โครงสร้างโปรเจกต์แนะนำ</h2>
      <pre className="bg-gray-900 text-white p-4 rounded-lg mt-2 overflow-x-auto text-sm">
        {`my-app/
├── node_modules/
├── routes/
│   └── userRoutes.js
├── server.js
├── package.json
└── .env`}
      </pre>

      <p className="mt-4 text-gray-600 dark:text-gray-300">
        ✨ Express.js เป็นเครื่องมือที่ทรงพลังในการเริ่มต้นสร้าง Backend อย่างง่ายและรวดเร็ว หากต้องการต่อยอดสามารถเรียนรู้เรื่อง Middleware, Router, และ Error Handling ต่อได้
      </p>
    </div>
  );
};

export default ExpressIntro;
