import React from "react";

const RestVersioning = () => {
  return (
    <div className="max-w-3xl mx-auto p-6 text-left">
      <h1 className="text-3xl font-bold mb-4">การจัดการเวอร์ชันของ REST API</h1>
      <p className="mb-4">
        ในการพัฒนา API เราอาจต้องมีการอัปเดตหรือเปลี่ยนแปลงโครงสร้าง API ซึ่งอาจทำให้ API รุ่นเก่าใช้งานไม่ได้
        ดังนั้นจึงต้องมี **การจัดการเวอร์ชัน (API Versioning)** เพื่อให้ API สามารถรองรับหลายเวอร์ชันได้พร้อมกัน  
      </p>

      <h2 className="text-2xl font-semibold mt-6 mb-2">📌 วิธีการจัดการเวอร์ชันของ API</h2>
      <ul className="list-disc pl-6 space-y-2">
        <li>**URI Versioning:** ระบุเวอร์ชันใน URL เช่น <code>/api/v1/users</code></li>
        <li>**Query Parameter Versioning:** ใช้พารามิเตอร์ใน URL เช่น <code>/api/users?version=1</code></li>
        <li>**Header Versioning:** กำหนดเวอร์ชันใน HTTP Header เช่น <code>Accept: application/vnd.myapi.v1+json</code></li>
      </ul>

      <h2 className="text-2xl font-semibold mt-6 mb-2">📌 ตัวอย่างการใช้งาน API Versioning</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto">
        <code>
{`const express = require("express");
const app = express();

app.get("/api/v1/users", (req, res) => {
  res.json({ version: "v1", users: [{ id: 1, name: "John" }] });
});

app.get("/api/v2/users", (req, res) => {
  res.json({ version: "v2", users: [{ id: 1, fullName: "John Doe" }] });
});

app.listen(3000, () => console.log("Server running on port 3000"));`}
        </code>
      </pre>

      <h2 className="text-2xl font-semibold mt-6 mb-2">📌 วิธีเลือกใช้ Versioning ที่เหมาะสม</h2>
      <p className="mb-4">
        วิธีที่ดีที่สุดขึ้นอยู่กับ **ความสะดวกของนักพัฒนา API และผู้ใช้ API**
      </p>
      <ul className="list-disc pl-6 space-y-2">
        <li><strong>URI Versioning</strong>: ง่ายต่อการใช้ แต่ทำให้ URL ยาวขึ้น</li>
        <li><strong>Query Parameter Versioning</strong>: ยืดหยุ่น แต่ไม่ใช่มาตรฐานที่ใช้กันทั่วไป</li>
        <li><strong>Header Versioning</strong>: ยืดหยุ่นมาก แต่ใช้งานยุ่งยากกว่า</li>
      </ul>
    </div>
  );
};

export default RestVersioning;
