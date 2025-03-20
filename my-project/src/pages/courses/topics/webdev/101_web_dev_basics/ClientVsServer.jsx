import React from "react";

const ClientVsServer = () => {
  return (
    <>
      <h1 className="text-2xl font-bold mb-4">Client vs Server</h1>
      <p>
        เว็บไซต์ทำงานโดยใช้โมเดล Client-Server ซึ่ง Client (เบราว์เซอร์) ทำการร้องขอข้อมูล และ Server จะประมวลผลและส่งคำตอบกลับมา
      </p>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">ฝั่ง Client</h2>
      <p>
        Client คือเว็บเบราว์เซอร์หรือแอปพลิเคชันที่ผู้ใช้โต้ตอบโดยตรง มีหน้าที่แสดงผล UI ส่งคำขอไปยังเซิร์ฟเวอร์ และจัดการข้อมูลที่ได้รับกลับมา
      </p>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: การดึงข้อมูลจากเซิร์ฟเวอร์</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`fetch('https://api.example.com/data')
  .then(response => response.json())
  .then(data => console.log(data));`}
      </pre>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">ฝั่ง Server</h2>
      <p>
        Server ทำหน้าที่จัดการคำขอ ประมวลผลข้อมูล และส่งข้อมูลกลับไปยัง Client ตัวอย่างเช่น การให้บริการ API หรือการจัดการฐานข้อมูล
      </p>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: เซิร์ฟเวอร์ Express.js อย่างง่าย</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`const express = require('express');
const app = express();

app.get('/data', (req, res) => {
  res.json({ message: 'Hello from the server!' });
});

app.listen(3000, () => console.log('Server running on port 3000'));`}
      </pre>
    </>
  );
};

export default ClientVsServer;