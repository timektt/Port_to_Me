import React from "react";

const ExpressMiddleware = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold">🛠️ Express.js Middleware</h1>

      <p className="mt-4">
        <strong>Middleware</strong> คือฟังก์ชันใน Express ที่ทำงานก่อน Route Handler โดยรับค่า <code>(req, res, next)</code> 
        และสามารถจัดการกับคำขอ, ตรวจสอบสิทธิ์, log, แปลงข้อมูล ฯลฯ ได้
      </p>

      {/* ✅ Custom Middleware */}
      <h2 className="text-xl font-semibold mt-6">🔹 Custom Middleware (สร้างเอง)</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`const express = require("express");
const app = express();

// Logger Middleware
const logger = (req, res, next) => {
  console.log(\`\${req.method} \${req.url}\`);
  next();
};

app.use(logger); // ใช้ middleware ทุก request

app.get("/", (req, res) => {
  res.send("Hello, Middleware!");
});

app.listen(3000, () => console.log("Server running on port 3000"));`}</code>
      </pre>

      {/* ✅ Built-in Middleware */}
      <h2 className="text-xl font-semibold mt-6">🔹 Built-in Middleware</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`// รับ JSON
app.use(express.json());

// รับข้อมูลแบบ Form (urlencoded)
app.use(express.urlencoded({ extended: true }));`}</code>
      </pre>
      <p className="mt-2">
        📌 ใช้เพื่อรับข้อมูลที่ส่งมาจาก frontend เช่น form หรือ API ที่ส่งแบบ JSON
      </p>

      {/* ✅ Third-party Middleware */}
      <h2 className="text-xl font-semibold mt-6">🔹 Third-party Middleware</h2>
      <p className="mt-2">Express รองรับการติดตั้ง Middleware เพิ่มเติม เช่น:</p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`const cors = require("cors");
const morgan = require("morgan");

app.use(cors());        // ป้องกันปัญหา Cross-Origin
app.use(morgan("dev")); // แสดง log คำขอ HTTP`}</code>
      </pre>

      {/* ✅ Error-handling Middleware */}
      <h2 className="text-xl font-semibold mt-6">⚠️ Error-handling Middleware</h2>
      <p className="mt-2">
        Middleware สำหรับจัดการ Error ต้องมีพารามิเตอร์ 4 ตัว: <code>(err, req, res, next)</code>
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: "Something went wrong!" });
});`}</code>
      </pre>

      <p className="mt-6">
        ✅ Middleware เป็นหัวใจหลักของ Express ที่ช่วยให้ระบบจัดการได้อย่างยืดหยุ่น เช่น Auth, Logging, การจัดการ Error, และการตรวจสอบ Token
      </p>
    </div>
  );
};

export default ExpressMiddleware;
