import React from "react";

const ApiGatewaysMicroservices = () => {
  return (
    <div className="max-w-4xl mx-auto p-6 text-left">
      <h1 className="text-3xl font-bold mb-4">🛡️ API Gateways & Microservices</h1>

      <p className="mb-4">
        <strong>API Gateway</strong> คือจุดเชื่อมกลางที่รับคำร้องขอทั้งหมดจาก Client 
        และส่งต่อคำร้องขอเหล่านั้นไปยัง Microservices ที่เกี่ยวข้องภายในระบบ โดยไม่ให้ผู้ใช้ต้องรู้ว่าเบื้องหลังมีกี่บริการ
      </p>

      <h2 className="text-2xl font-semibold mt-6">📌 ทำไมต้องใช้ API Gateway?</h2>
      <ul className="list-disc pl-6 space-y-2 text-gray-700 dark:text-gray-300">
        <li>ควบคุมเส้นทางและความปลอดภัยของ API ทั้งหมดในที่เดียว</li>
        <li>รวมการตรวจสอบสิทธิ์ (Authentication)</li>
        <li>ทำ Rate Limiting, Caching และ Logging ได้</li>
        <li>ช่วยให้ Microservices ทำงานได้อย่างอิสระ</li>
      </ul>

      <h2 className="text-xl font-semibold mt-6">📎 ตัวอย่างโครงสร้าง API Gateway ด้วย Express</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        <code>{`// gateway.js
const express = require("express");
const app = express();

// เชื่อมต่อไปยัง Microservices
app.use("/user", require("./userService"));
app.use("/order", require("./orderService"));

app.listen(3000, () => console.log("🚀 API Gateway Running on port 3000"));`}</code>
      </pre>

      <p className="mt-4 text-gray-700 dark:text-gray-300">
        🔁 เช่น หากผู้ใช้ส่งคำร้องขอไปยัง <code>/user/profile</code> ระบบจะส่งต่อไปยัง <code>userService</code>
      </p>

      <h2 className="text-xl font-semibold mt-6">🧩 ตัวอย่าง userService</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        <code>{`// userService.js
const express = require("express");
const router = express.Router();

router.get("/profile", (req, res) => {
  res.json({ id: 1, name: "John Doe" });
});

module.exports = router;`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">📦 เหมาะสำหรับสถาปัตยกรรม Microservices</h2>
      <p className="mt-2 text-gray-700 dark:text-gray-300">
        สามารถขยายระบบได้อย่างอิสระ เช่นเพิ่มบริการ <code>/payment</code>, <code>/inventory</code> ได้ในอนาคต
      </p>
    </div>
  );
};

export default ApiGatewaysMicroservices;
