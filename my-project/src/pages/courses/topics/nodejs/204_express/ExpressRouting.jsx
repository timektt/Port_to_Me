import React from "react";

const ExpressRouting = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold">📍 Express.js Routing</h1>

      <p className="mt-4">
        Routing ใน Express ใช้สำหรับกำหนดเส้นทางที่ผู้ใช้สามารถเข้าถึง โดยใช้ HTTP Methods เช่น <code>GET</code>, <code>POST</code>, <code>PUT</code>, <code>DELETE</code>
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่างการกำหนด Route</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`const express = require("express");
const app = express();

app.get("/", (req, res) => res.send("Home Page"));
app.get("/about", (req, res) => res.send("About Page"));
app.get("/contact", (req, res) => res.send("Contact Page"));

app.listen(3000, () => console.log("Server running on port 3000"));`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">📌 Dynamic Route (Parameter)</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`app.get("/user/:id", (req, res) => {
  res.send(\`User ID: \${req.params.id}\`);
});`}</code>
      </pre>
      <p className="mt-2">🔎 ทดสอบที่ <code>http://localhost:3000/user/123</code></p>

      <h2 className="text-xl font-semibold mt-6">📌 การใช้ HTTP Methods อื่น ๆ</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`app.post("/submit", (req, res) => {
  res.send("POST Request Received");
});

app.put("/update/:id", (req, res) => {
  res.send(\`Updated user \${req.params.id}\`);
});

app.delete("/delete/:id", (req, res) => {
  res.send(\`Deleted user \${req.params.id}\`);
});`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">📌 แยก Routing ออกเป็นไฟล์ (Router)</h2>
      <p className="mt-2">เพื่อความเป็นระเบียบในโปรเจกต์ใหญ่</p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`// routes/userRoutes.js
const express = require("express");
const router = express.Router();

router.get("/", (req, res) => res.send("User List"));
router.get("/:id", (req, res) => res.send(\`User ID: \${req.params.id}\`));

module.exports = router;`}</code>
      </pre>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`// app.js
const express = require("express");
const app = express();
const userRoutes = require("./routes/userRoutes");

app.use("/users", userRoutes);

app.listen(3000, () => console.log("Server running"));`}</code>
      </pre>

      <p className="mt-6">
        ✅ การจัดการ Routing ที่ดีช่วยให้โครงสร้างโปรเจกต์ของคุณสะอาดและขยายได้ง่าย
      </p>
    </div>
  );
};

export default ExpressRouting;
