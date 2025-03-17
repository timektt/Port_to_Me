import React from "react";

const ExpressErrorHandling = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold">⚠️ Express.js Error Handling</h1>
      <p className="mt-4">
        Express มีวิธีจัดการข้อผิดพลาดผ่าน Middleware เพื่อป้องกันการทำงานที่ไม่พึงประสงค์
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่าง Error Handling Middleware</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2">
        <code>{`const express = require("express");
const app = express();

app.get("/error", (req, res, next) => {
  next(new Error("Something went wrong!"));
});

// Middleware จัดการ Error
app.use((err, req, res, next) => {
  res.status(500).json({ message: err.message });
});

app.listen(3000, () => console.log("Server running on port 3000"));`}</code>
      </pre>

      <p className="mt-4">🔥 ทดสอบโดยเรียก **http://localhost:3000/error**</p>
    </div>
  );
};

export default ExpressErrorHandling;
