import React from "react";

const ExpressErrorHandling = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold">⚠️ Express.js Error Handling</h1>

      <p className="mt-4">
        Express.js รองรับการจัดการข้อผิดพลาดผ่าน <strong>Middleware</strong> โดยเฉพาะ ซึ่งช่วยให้แยก logic สำหรับ error ออกจาก route logic ได้อย่างชัดเจน
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 Basic Error Middleware</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto text-sm">
        <code>{`const express = require("express");
const app = express();

app.get("/error", (req, res, next) => {
  // จำลองการเกิดข้อผิดพลาด
  next(new Error("Something went wrong!"));
});

// Middleware สำหรับจัดการ Error
app.use((err, req, res, next) => {
  console.error(err.stack); // แสดง Stack Trace
  res.status(500).json({ message: err.message || "Internal Server Error" });
});

app.listen(3000, () => console.log("🚀 Server running on port 3000"));`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">✅ การใช้งานในโปรเจกต์จริง</h2>
      <ul className="list-disc ml-6 mt-2 space-y-1">
        <li>แนะนำให้เขียน error middleware ไว้ <strong>ท้ายสุดของ App</strong></li>
        <li>สามารถเพิ่ม Custom Error Class เพื่อแยกประเภทของ Error</li>
        <li>อย่าลืม log ข้อผิดพลาดไว้ด้วย เช่นใช้ <code>winston</code>, <code>pino</code>, หรือ log ธรรมดา</li>
        <li>ใน Production ควรซ่อนข้อความ error ที่ละเอียด เพื่อความปลอดภัย</li>
      </ul>

      <h2 className="text-xl font-semibold mt-6">🛡️ ตัวอย่าง Custom Error Handler</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto text-sm">
        <code>{`app.use((err, req, res, next) => {
  const status = err.status || 500;
  res.status(status).json({
    success: false,
    error: {
      message: err.message,
      status: status
    }
  });
});`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">🧪 วิธีทดสอบ</h2>
      <p className="mt-2">1. สร้าง endpoint ที่โยน error โดยตรง</p>
      <p className="mt-1">2. ใช้ Postman หรือเบราว์เซอร์เรียก <code>http://localhost:3000/error</code></p>
    </div>
  );
};

export default ExpressErrorHandling;
