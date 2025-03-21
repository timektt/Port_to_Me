import React from "react";

const JwtSessionManagement = () => {
  return (
    <div className="max-w-4xl mx-auto p-6 text-left">
      <h1 className="text-3xl font-bold mb-4 text-gray-900 dark:text-white">
        🔐 JWT & Session Management
      </h1>

      <p className="mb-4 text-gray-700 dark:text-gray-300">
        <strong>JWT (JSON Web Token)</strong> คือ Token ที่ใช้ตรวจสอบตัวตนของผู้ใช้ โดยฝังข้อมูลลงไปใน Token แบบเข้ารหัส
        และสามารถใช้แบบ Stateless ได้ ต่างจาก Session แบบเดิมที่ต้องเก็บใน Server
      </p>

      <h2 className="text-xl font-semibold mt-6 text-gray-800 dark:text-gray-200">📌 ตัวอย่างการสร้าง JWT Token</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        <code>{`const jwt = require("jsonwebtoken");

const user = { id: 1, name: "John Doe" };

// สร้าง Token โดยใช้ secretKey และกำหนดเวลาหมดอายุ 1 ชั่วโมง
const token = jwt.sign(user, "secretKey", { expiresIn: "1h" });

console.log("JWT Token:", token);`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6 text-gray-800 dark:text-gray-200">📌 การตรวจสอบ JWT Token</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        <code>{`const token = "TOKEN_FROM_CLIENT";

try {
  const decoded = jwt.verify(token, "secretKey");
  console.log("Decoded Data:", decoded);
} catch (err) {
  console.error("Invalid Token:", err.message);
}`}</code>
      </pre>

      <div className="mt-6 p-4 bg-blue-100 dark:bg-blue-900 text-blue-900 dark:text-blue-200 rounded-lg">
        💡 <strong>Note:</strong> ควรเก็บ Secret Key ไว้ใน Environment Variables และใช้ HTTPS เพื่อป้องกัน Token ถูกดัก
      </div>
    </div>
  );
};

export default JwtSessionManagement;
