import React from "react";

const RestVersioning = () => {
  return (
    <div className="max-w-3xl mx-auto p-6 text-left">
      <h1 className="text-3xl font-bold mb-4">🧩 การจัดการเวอร์ชันของ REST API</h1>

      <p className="mb-4">
        ในการพัฒนา API เราอาจต้องมีการอัปเดตหรือเปลี่ยนแปลงโครงสร้าง API ซึ่งอาจทำให้ API รุ่นเก่าใช้งานไม่ได้
        ดังนั้นจึงต้องมี <strong>การจัดการเวอร์ชัน (API Versioning)</strong> เพื่อให้สามารถรองรับหลายเวอร์ชันได้พร้อมกัน
      </p>

      <h2 className="text-2xl font-semibold mt-6 mb-2">📌 วิธีการจัดการเวอร์ชันของ API</h2>
      <ul className="list-disc pl-6 space-y-2">
        <li><strong>URI Versioning:</strong> <code>/api/v1/users</code></li>
        <li><strong>Query Parameter Versioning:</strong> <code>/api/users?version=1</code></li>
        <li><strong>Header Versioning:</strong> <code>Accept: application/vnd.myapi.v1+json</code></li>
      </ul>

      <h2 className="text-2xl font-semibold mt-6 mb-2">📌 ตัวอย่างโค้ด Express.js</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto text-sm">
        <code>{`const express = require("express");
const app = express();

// URI Versioning
app.get("/api/v1/users", (req, res) => {
  res.json({ version: "v1", users: [{ id: 1, name: "John" }] });
});

app.get("/api/v2/users", (req, res) => {
  res.json({ version: "v2", users: [{ id: 1, fullName: "John Doe" }] });
});

app.listen(3000, () => console.log("Server running on port 3000"));`}</code>
      </pre>

      <h2 className="text-2xl font-semibold mt-6 mb-2">📌 คำแนะนำในการเลือกวิธี Versioning</h2>
      <p className="mb-2">
        ควรเลือกตามความเหมาะสมกับระบบและผู้ใช้งาน API ของคุณ:
      </p>
      <ul className="list-disc pl-6 space-y-2">
        <li><strong>URI Versioning:</strong> ใช้งานง่าย เห็นชัดเจน เหมาะกับ public APIs</li>
        <li><strong>Query Parameter:</strong> ยืดหยุ่นแต่ไม่ค่อยนิยม</li>
        <li><strong>Header Versioning:</strong> เหมาะกับ API ที่ต้องการควบคุมละเอียด แต่ยากต่อการทดสอบด้วยเบราว์เซอร์</li>
      </ul>
    </div>
  );
};

export default RestVersioning;
