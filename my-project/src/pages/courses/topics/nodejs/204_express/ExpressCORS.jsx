import React from "react";

const ExpressCORS = () => {
  return (
    <div className="max-w-3xl mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">🌐 Express.js & CORS</h1>

      <p className="mb-4">
        <strong>CORS (Cross-Origin Resource Sharing)</strong> คือกลไกที่ช่วยให้เบราว์เซอร์อนุญาตให้เว็บจากโดเมนหนึ่งเข้าถึงทรัพยากรจากอีกโดเมนหนึ่งได้
        ซึ่งโดยปกติแล้วเบราว์เซอร์จะไม่อนุญาตให้ทำแบบนี้เพื่อความปลอดภัย
      </p>

      <h2 className="text-xl font-semibold mt-6 mb-2">📌 ปัญหา: CORS Error</h2>
      <p className="mb-4">
        เมื่อเว็บ frontend พยายามเรียก API ข้ามโดเมน เช่น จาก <code>http://localhost:3000</code> ไปยัง <code>http://localhost:5000</code>
        โดยไม่มีการตั้งค่า CORS จะเกิดข้อความ "Blocked by CORS policy"
      </p>

      <h2 className="text-xl font-semibold mt-6 mb-2">✅ วิธีแก้: ติดตั้ง CORS Middleware</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mb-4">
        <code>npm install cors</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6 mb-2">✅ การใช้งานใน Express.js</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto text-sm mb-4">
        <code>{`const express = require("express");
const cors = require("cors");
const app = express();

// เปิดใช้งาน CORS แบบเปิดกว้าง (Allow All Origins)
app.use(cors());

// หรือจำกัดเฉพาะบาง origin
// app.use(cors({ origin: "http://localhost:3000" }));

app.get("/api/data", (req, res) => {
  res.json({ message: "CORS Enabled API" });
});

app.listen(5000, () => console.log("Server running on port 5000"));`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6 mb-2">🔐 ข้อควรระวัง</h2>
      <ul className="list-disc ml-5 text-gray-800 dark:text-gray-300">
        <li>หลีกเลี่ยงการเปิดกว้าง <code>origin: "*"</code> ใน production</li>
        <li>สามารถกำหนด origin แบบ whitelist เพื่อเพิ่มความปลอดภัย</li>
        <li>ตั้งค่าการอนุญาต method, headers, credentials ได้เพิ่มเติม</li>
      </ul>

      <h2 className="text-xl font-semibold mt-6 mb-2">🧪 ทดสอบ</h2>
      <p>
        ลองเรียก API นี้จาก frontend (React, Vue, ฯลฯ) แล้วเช็คว่าไม่มี CORS Error อีก
      </p>
    </div>
  );
};

export default ExpressCORS;
