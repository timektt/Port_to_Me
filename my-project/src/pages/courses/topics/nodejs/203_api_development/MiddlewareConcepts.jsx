import React from "react";

const MiddlewareConcepts = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto overflow-x-hidden">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold break-words">
        🛠️ Middleware Concepts
      </h1>

      <p className="mt-4 text-lg break-words">
        Middleware เป็นฟังก์ชันที่ทำงานก่อนที่ Request จะไปถึง Route Handler เช่น{" "}
        <strong>Logging, Authentication, Error Handling</strong>
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่าง Middleware</h2>
      <div className="overflow-x-auto">
        <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 text-sm sm:text-base">
          <code>{`const express = require("express");
const app = express();

const logger = (req, res, next) => {
  console.log(\`\${req.method} \${req.url}\`);
  next();
};

app.use(logger);

app.get("/", (req, res) => {
  res.send("Hello, Middleware!");
});

app.listen(3000, () => console.log("Server running on port 3000"));`}</code>
        </pre>
      </div>

      <p className="mt-4 text-base">⚡ ลองรันแล้วเช็ค <code>console.log()</code></p>

      <h2 className="text-xl font-semibold mt-6">🔐 Middleware สำหรับ Authentication</h2>
      <p className="mt-2 text-base break-words">
        สามารถใช้ Middleware เพื่อตรวจสอบ Token หรือสิทธิ์ของผู้ใช้งาน ก่อนเข้าถึง API ที่ต้องการความปลอดภัย
      </p>
      <div className="overflow-x-auto">
        <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 text-sm sm:text-base">
          <code>{`const auth = (req, res, next) => {
  const token = req.headers["authorization"];
  if (token === "secret") {
    next();
  } else {
    res.status(401).json({ message: "Unauthorized" });
  }
};

app.get("/secure", auth, (req, res) => {
  res.send("Access granted!");
});`}</code>
        </pre>
      </div>

      <h2 className="text-xl font-semibold mt-6">🧼 การจัดลำดับ Middleware</h2>
      <p className="mt-2 text-base break-words">
        Middleware จะทำงานตามลำดับที่ <code>app.use()</code> หรือ <code>app.get()</code> ถูกเรียก ดังนั้นควรจัดลำดับให้เหมาะสม เช่น Logging → Auth → Handler
      </p>

      <h2 className="text-xl font-semibold mt-6">🧩 ประเภทของ Middleware</h2>
      <ul className="list-disc ml-6 mt-2 space-y-1 text-base">
        <li><strong>Application-level middleware</strong> - ใช้กับ <code>app.use()</code></li>
        <li><strong>Router-level middleware</strong> - ใช้กับ <code>express.Router()</code></li>
        <li><strong>Error-handling middleware</strong> - ฟังก์ชันที่มี <code>(err, req, res, next)</code></li>
        <li><strong>Built-in middleware</strong> - เช่น <code>express.json()</code>, <code>express.static()</code></li>
        <li><strong>Third-party middleware</strong> - เช่น <code>cors</code>, <code>morgan</code></li>
      </ul>

      <h2 className="text-xl font-semibold mt-6">✅ สรุป</h2>
      <p className="mt-2 text-base break-words">
        Middleware เป็นเครื่องมือสำคัญใน Express ที่ช่วยจัดการ Request/Response ได้ยืดหยุ่นมากขึ้น
        ใช้สำหรับตรวจสอบข้อมูล ล็อกข้อมูล หรือแม้แต่ควบคุมสิทธิ์การเข้าถึง API
      </p>
    </div>
  );
};

export default MiddlewareConcepts;
