import React from "react";

const FrontendVsBackend = () => {
  return (
    <>
      <h1 className="text-2xl font-bold mb-4">Frontend vs Backend</h1>
      <p>
        ในการพัฒนาเว็บไซต์ ระบบสามารถแบ่งออกเป็นสองส่วนหลัก ได้แก่ Frontend และ Backend ซึ่งแต่ละส่วนมีหน้าที่และทักษะที่จำเป็นแตกต่างกัน
      </p>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">Frontend Development</h2>
      <p>
        Frontend หรือฝั่งผู้ใช้ (Client-side) คือส่วนที่ผู้ใช้โต้ตอบโดยตรง ซึ่งรวมถึง HTML, CSS และ JavaScript รวมถึงเฟรมเวิร์กอย่าง React, Vue หรือ Angular
      </p>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: ปุ่ม HTML & CSS อย่างง่าย</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`<button style="background-color: blue; color: white; padding: 10px; border: none;">
  Click Me
</button>`}
      </pre>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">Backend Development</h2>
      <p>
        Backend หรือฝั่งเซิร์ฟเวอร์ (Server-side) ทำหน้าที่ประมวลผลข้อมูล การยืนยันตัวตน และการจัดการฐานข้อมูล เทคโนโลยีที่ใช้ทั่วไป เช่น Node.js, Python, MongoDB และ PostgreSQL
      </p>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: เซิร์ฟเวอร์ Node.js อย่างง่าย</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`const express = require('express');
const app = express();

app.get('/', (req, res) => {
  res.send('Hello from the backend!');
});

app.listen(3000, () => console.log('Server running on port 3000'));`}
      </pre>
    </>
  );
};

export default FrontendVsBackend;