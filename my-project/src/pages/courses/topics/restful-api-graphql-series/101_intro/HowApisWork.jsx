import React from "react";

const HowApisWork = () => {
  return (
    <div className="max-w-3xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-4">🔌 การทำงานของ API</h1>

      <p className="text-lg">
        API (Application Programming Interface) ทำงานโดยเป็นสื่อกลางระหว่าง <strong>Client</strong> (เช่น Web หรือ Mobile App) 
        กับ <strong>Server</strong> (ระบบเบื้องหลังที่เก็บข้อมูลหรือประมวลผล)
      </p>

      <h2 className="text-xl font-semibold mt-6">📥 ขั้นตอนการทำงาน</h2>
      <ol className="list-decimal ml-6 mt-2 space-y-2 text-gray-700 dark:text-gray-300">
        <li>Client ส่งคำร้องขอ (Request) ไปยัง API endpoint</li>
        <li>API รับคำขอ และส่งต่อให้ Server ประมวลผล</li>
        <li>Server ประมวลผลคำขอ เช่น ดึงข้อมูลจากฐานข้อมูล หรือคำนวณ</li>
        <li>API ส่งผลลัพธ์กลับไปยัง Client ในรูปแบบ JSON หรือ XML</li>
      </ol>

      <h2 className="text-xl font-semibold mt-6">🌐 ตัวอย่าง HTTP Methods</h2>
      <ul className="list-disc ml-6 mt-2 space-y-1">
        <li><strong>GET</strong>: ใช้สำหรับดึงข้อมูล เช่น รายชื่อผู้ใช้</li>
        <li><strong>POST</strong>: ใช้สำหรับสร้างข้อมูลใหม่ เช่น เพิ่มผู้ใช้ใหม่</li>
        <li><strong>PUT</strong>: ใช้สำหรับแก้ไขข้อมูลที่มีอยู่</li>
        <li><strong>DELETE</strong>: ใช้สำหรับลบข้อมูล</li>
      </ul>

      <div className="bg-gray-800 text-white p-4 mt-6 rounded-lg overflow-x-auto text-sm">
        <pre>{`GET /api/users HTTP/1.1
Host: example.com

Response:
HTTP/1.1 200 OK
Content-Type: application/json

[
  { "id": 1, "name": "Alice" },
  { "id": 2, "name": "Bob" }
]`}</pre>
      </div>

      <div className="mt-6 p-4 bg-blue-100 dark:bg-blue-900 text-blue-900 dark:text-blue-300 rounded-lg shadow">
        💡 <strong>สรุป:</strong> API ช่วยให้ระบบต่าง ๆ สื่อสารกันได้ผ่านภาษากลาง โดยไม่จำเป็นต้องรู้โครงสร้างภายในของกันและกัน
      </div>
    </div>
  );
};

export default HowApisWork;
