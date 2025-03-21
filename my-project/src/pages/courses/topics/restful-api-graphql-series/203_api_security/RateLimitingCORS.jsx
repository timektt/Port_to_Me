import React from "react";

const RateLimitingCORS = () => {
  return (
    <div className="max-w-4xl mx-auto p-6 text-left">
      <h1 className="text-3xl font-bold mb-4 text-gray-900 dark:text-white">🛡️ Rate Limiting & CORS</h1>

      {/* Rate Limiting Section */}
      <p className="mb-4 text-gray-700 dark:text-gray-300">
        <strong>Rate Limiting</strong> คือการจำกัดจำนวนคำขอที่สามารถส่งไปยัง API ได้ในระยะเวลาหนึ่ง เพื่อป้องกันการโจมตีแบบ DDoS และลดภาระเซิร์ฟเวอร์
      </p>

      <h2 className="text-xl font-semibold mt-6 text-gray-800 dark:text-gray-200">📌 ติดตั้งและใช้งาน express-rate-limit</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        <code>{`// ติดตั้ง: npm install express-rate-limit
const rateLimit = require("express-rate-limit");

const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 นาที
  max: 100, // จำกัด 100 requests ต่อ IP
  message: "Too many requests, please try again later."
});

app.use(limiter);`}</code>
      </pre>

      {/* CORS Section */}
      <h2 className="text-xl font-semibold mt-8 text-gray-800 dark:text-gray-200">🌐 CORS (Cross-Origin Resource Sharing)</h2>
      <p className="mt-2 text-gray-700 dark:text-gray-300">
        <strong>CORS</strong> เป็นกลไกด้านความปลอดภัยที่จำกัดการเข้าถึง API จากโดเมนอื่น หากต้องการให้เว็บไซต์อื่นเข้าถึง API ได้ต้องเปิด CORS
      </p>

      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-4">
        <code>{`// ติดตั้ง: npm install cors
const cors = require("cors");

// อนุญาตให้ทุกโดเมนเรียกใช้ API
app.use(cors());`}</code>
      </pre>

      <p className="mt-4 text-sm text-yellow-600 dark:text-yellow-400">
        ⚠️ แนะนำให้ตั้งค่าต้นทางที่อนุญาตแบบเจาะจง เช่น:
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-2">
        <code>{`app.use(cors({
  origin: ["https://your-frontend.com"],
  methods: ["GET", "POST"],
  credentials: true
}));`}</code>
      </pre>

      {/* Summary */}
      <div className="mt-6 p-4 bg-blue-100 dark:bg-blue-900 text-blue-900 dark:text-blue-200 rounded-lg">
        💡 <strong>สรุป:</strong> ใช้ <code>express-rate-limit</code> เพื่อควบคุมการใช้งาน API และ <code>cors</code> เพื่อกำหนดว่าใครสามารถเข้าถึง API ได้
      </div>
    </div>
  );
};

export default RateLimitingCORS;
