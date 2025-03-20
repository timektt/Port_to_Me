import React from "react";

const NodeExpress = () => {
  return (
    <>
      <h1 className="text-2xl font-bold mb-4">Node.js & Express</h1>
      <p>
        Node.js เป็น JavaScript runtime ที่ช่วยให้สามารถพัฒนา Backend ได้โดยใช้ JavaScript ส่วน Express เป็น framework ที่ใช้สำหรับสร้าง Web Server ใน Node.js ได้ง่ายขึ้น
      </p>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">คุณสมบัติหลักของ Node.js</h2>
      <ul className="list-disc pl-6 mb-4">
        <li>ทำงานแบบ asynchronous และ non-blocking</li>
        <li>รองรับการทำงานแบบ event-driven</li>
        <li>มี npm (Node Package Manager) สำหรับจัดการ dependencies</li>
      </ul>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">คุณสมบัติหลักของ Express</h2>
      <ul className="list-disc pl-6 mb-4">
        <li>รองรับ Middleware สำหรับจัดการ HTTP Requests</li>
        <li>สามารถกำหนดเส้นทาง (Routing) ได้ง่าย</li>
        <li>รองรับการใช้งานกับ Database เช่น MongoDB, MySQL</li>
      </ul>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: การสร้าง Server ด้วย Express</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`const express = require('express');
const app = express();

app.get('/', (req, res) => {
  res.send('Hello from Express!');
});

app.listen(3000, () => {
  console.log('Server running on port 3000');
});`}
      </pre>
    </>
  );
};

export default NodeExpress;
