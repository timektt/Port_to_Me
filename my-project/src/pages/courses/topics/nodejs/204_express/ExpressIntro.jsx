import React from "react";

const ExpressIntro = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold">🚀 Introduction to Express.js</h1>
      <p className="mt-4">
        Express.js เป็น Framework ที่ช่วยให้การพัฒนา Web Application และ API ด้วย Node.js ง่ายขึ้น
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 วิธีติดตั้ง Express.js</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2">
        <code>{`npm init -y
npm install express`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่างโค้ด Express.js เบื้องต้น</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2">
        <code>{`const express = require("express");
const app = express();

app.get("/", (req, res) => {
  res.send("Hello, Express!");
});

app.listen(3000, () => console.log("Server running on port 3000"));`}</code>
      </pre>

      <p className="mt-4">✅ เปิดใช้งานที่ **http://localhost:3000/**</p>
    </div>
  );
};

export default ExpressIntro;
