import React from "react";

const RestGraphQL = () => {
  return (
    <>
      <h1 className="text-2xl font-bold mb-4">REST & GraphQL APIs</h1>
      <p>
        REST และ GraphQL เป็นแนวทางที่นิยมใช้ในการพัฒนา API โดย REST ใช้โครงสร้างแบบ Resource-based และ GraphQL ใช้ Query-based
      </p>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">REST API</h2>
      <p>
        REST (Representational State Transfer) เป็นมาตรฐานสำหรับการพัฒนา API โดยใช้ HTTP methods เช่น GET, POST, PUT และ DELETE
      </p>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: REST API ด้วย Express.js</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`const express = require('express');
const app = express();

app.get('/users', (req, res) => {
  res.json([{ id: 1, name: 'Alice' }, { id: 2, name: 'Bob' }]);
});

app.listen(3000, () => console.log('API running on port 3000'));`}
      </pre>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">GraphQL API</h2>
      <p>
        GraphQL เป็นเทคโนโลยีที่ช่วยให้ไคลเอนต์สามารถกำหนดข้อมูลที่ต้องการได้อย่างยืดหยุ่นผ่าน Query Language
      </p>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: GraphQL Query</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`{
  user(id: 1) {
    name
    age
  }
}`}
      </pre>
    </>
  );
};

export default RestGraphQL;
