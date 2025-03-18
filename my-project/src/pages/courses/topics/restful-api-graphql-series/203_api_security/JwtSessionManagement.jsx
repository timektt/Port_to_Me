import React from "react";

const JwtSessionManagement = () => {
  return (
    <div className="max-w-4xl mx-auto p-6 text-left">
      <h1 className="text-3xl font-bold mb-4">JWT & Session Management</h1>
      <p className="mb-4">
        <strong>JWT (JSON Web Token)</strong> ใช้สำหรับตรวจสอบตัวตนของผู้ใช้โดยการเข้ารหัสข้อมูลที่จำเป็นลงใน token.
      </p>
      <h2 className="text-xl font-semibold mt-4">ตัวอย่างการสร้าง JWT Token</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg">
        {`const jwt = require("jsonwebtoken");

const user = { id: 1, name: "John Doe" };
const token = jwt.sign(user, "secretKey", { expiresIn: "1h" });

console.log(token);`}
      </pre>
    </div>
  );
};

export default JwtSessionManagement;
