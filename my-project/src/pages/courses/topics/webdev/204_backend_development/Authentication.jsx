import React from "react";

const Authentication = () => {
  return (
    <>
      <h1 className="text-2xl font-bold mb-4">การยืนยันตัวตน & การอนุญาต (Authentication & Authorization)</h1>
      <p>
        Authentication เป็นกระบวนการตรวจสอบตัวตนของผู้ใช้ ในขณะที่ Authorization เป็นการกำหนดสิทธิ์การเข้าถึงของผู้ใช้
      </p>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">วิธีการยืนยันตัวตนยอดนิยม</h2>
      <ul className="list-disc pl-6 mb-4">
        <li>JWT (JSON Web Token)</li>
        <li>OAuth 2.0</li>
        <li>Session-based Authentication</li>
      </ul>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: Authentication ด้วย JWT</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`const jwt = require('jsonwebtoken');
const user = { id: 1, username: 'Alice' };
const token = jwt.sign(user, 'secretkey', { expiresIn: '1h' });
console.log(token);`}
      </pre>
    </>
  );
};

export default Authentication;
