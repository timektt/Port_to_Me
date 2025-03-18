import React from "react";

const RateLimitingCORS = () => {
  return (
    <div className="max-w-4xl mx-auto p-6 text-left">
      <h1 className="text-3xl font-bold mb-4">Rate Limiting & CORS</h1>
      <p className="mb-4">
        <strong>Rate Limiting</strong> เป็นเทคนิคที่ใช้ป้องกัน API จากการเรียกใช้งานมากเกินไป เพื่อป้องกันการโจมตี DDoS.
      </p>
      <h2 className="text-xl font-semibold mt-4">ติดตั้ง `express-rate-limit`</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg">
        {`const rateLimit = require("express-rate-limit");

const limiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 100
});

app.use(limiter);`}
      </pre>

      <h2 className="text-xl font-semibold mt-4">CORS (Cross-Origin Resource Sharing)</h2>
      <p className="mt-2">ใช้ `cors` middleware เพื่ออนุญาตให้เว็บอื่นเข้าถึง API ได้</p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg">
        {`const cors = require("cors");
app.use(cors());`}
      </pre>
    </div>
  );
};

export default RateLimitingCORS;
