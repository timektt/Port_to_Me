import React from "react";

const APIDevelopment = () => {
  return (
    <>
      <h1 className="text-2xl font-bold mb-4">การพัฒนา API</h1>
      <p>
        API (Application Programming Interface) เป็นกลไกที่ช่วยให้แอปพลิเคชันต่าง ๆ สามารถสื่อสารกันได้ โดยใช้โปรโตคอล HTTP หรือ WebSockets
      </p>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">ประเภทของ API</h2>
      <ul className="list-disc pl-6 mb-4">
        <li>RESTful API - อิงตามหลักการของ REST (Representational State Transfer)</li>
        <li>GraphQL API - ใช้แนวคิดของ Query-based API</li>
        <li>WebSockets API - ใช้สำหรับการสื่อสารแบบเรียลไทม์</li>
      </ul>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">ตัวอย่าง: การสร้าง RESTful API ด้วย Express</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`const express = require('express');
const app = express();
app.use(express.json());

const users = [{ id: 1, name: 'Alice' }, { id: 2, name: 'Bob' }];

app.get('/users', (req, res) => {
  res.json(users);
});

app.listen(3000, () => console.log('API Server running on port 3000'));`}
      </pre>
    </>
  );
};

export default APIDevelopment;
