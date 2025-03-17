import React from "react";

const HandlingHttpRequests = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold">📡 Handling HTTP Requests</h1>
      <p className="mt-4">
        Express.js รองรับการรับ-ส่งข้อมูลผ่าน HTTP Methods เช่น **GET, POST, PUT, DELETE**
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่างการรับค่า POST Request</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2">
        <code>{`const express = require("express");
const app = express();
app.use(express.json());

app.post("/api/users", (req, res) => {
  const { name } = req.body;
  res.json({ message: \`User \${name} created!\` });
});

app.listen(3000, () => console.log("Server running on port 3000"));`}</code>
      </pre>

      <p className="mt-4">🛠️ ทดสอบโดยใช้ **Postman หรือ cURL**</p>
    </div>
  );
};

export default HandlingHttpRequests;
