import React from "react";

const HandlingHttpRequests = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto overflow-x-hidden">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold break-words">
        📡 Handling HTTP Requests
      </h1>

      <p className="mt-4 text-lg break-words">
        Express.js รองรับการรับ-ส่งข้อมูลผ่าน HTTP Methods เช่น{" "}
        <strong>GET, POST, PUT, DELETE</strong>
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่างการรับค่า POST Request</h2>
      <div className="overflow-x-auto">
        <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 text-sm sm:text-base">
          <code>{`const express = require("express");
const app = express();
app.use(express.json());

app.post("/api/users", (req, res) => {
  const { name } = req.body;
  res.json({ message: \`User \${name} created!\` });
});

app.listen(3000, () => console.log("Server running on port 3000"));`}</code>
        </pre>
      </div>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่าง GET, PUT และ DELETE</h2>
      <div className="overflow-x-auto">
        <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 text-sm sm:text-base">
          <code>{`// GET - ดึงข้อมูลผู้ใช้ทั้งหมด
app.get("/api/users", (req, res) => {
  res.json([{ id: 1, name: "Alice" }, { id: 2, name: "Bob" }]);
});

// PUT - อัปเดตข้อมูลผู้ใช้
app.put("/api/users/:id", (req, res) => {
  const { id } = req.params;
  const { name } = req.body;
  res.json({ message: \`User \${id} updated to \${name}\` });
});

// DELETE - ลบผู้ใช้
app.delete("/api/users/:id", (req, res) => {
  const { id } = req.params;
  res.json({ message: \`User \${id} deleted\` });
});`}</code>
        </pre>
      </div>

      <p className="mt-4 text-base break-words">
        🛠️ ทดสอบ API เหล่านี้โดยใช้ <strong>Postman</strong> หรือ <strong>cURL</strong> สำหรับการจำลองคำขอ
      </p>

      <h2 className="text-xl font-semibold mt-6">🧠 หมายเหตุเพิ่มเติม</h2>
      <ul className="list-disc ml-6 mt-2 space-y-1 text-base">
        <li>ใช้ <code>express.json()</code> เพื่อแปลง JSON request body</li>
        <li>ใช้ <code>req.params</code> สำหรับดึงค่าใน URL เช่น <code>:id</code></li>
        <li>ใช้ <code>req.body</code> สำหรับดึงค่าจาก request body</li>
      </ul>
    </div>
  );
};

export default HandlingHttpRequests;
