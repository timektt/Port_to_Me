import React from "react";

const ApiPerformanceOptimization = () => {
  return (
    <div className="max-w-4xl mx-auto p-6 text-left">
      <h1 className="text-3xl font-bold mb-4">⚡ API Performance Optimization</h1>
      <p className="mb-4">
        การปรับปรุงประสิทธิภาพของ API เป็นสิ่งสำคัญ เพื่อให้รองรับผู้ใช้งานจำนวนมาก 
        และลดความหน่วงในการโหลดข้อมูล
      </p>

      <h2 className="text-2xl font-semibold mt-6 mb-2">📌 เทคนิคที่ช่วยให้ API ทำงานเร็วขึ้น</h2>
      <ul className="list-disc pl-6 space-y-2 text-gray-800 dark:text-gray-300">
        <li>✅ ใช้ Caching เช่น <code>Redis</code> เพื่อเก็บข้อมูลที่เรียกบ่อย</li>
        <li>✅ ใช้ GZIP Compression เพื่อลดขนาด response</li>
        <li>✅ ใช้ Load Balancer เช่น NGINX หรือ HAProxy</li>
        <li>✅ Index ตารางในฐานข้อมูลให้เหมาะสม</li>
        <li>✅ Limit และ Paginate ข้อมูล (ไม่ส่งข้อมูลทั้งหมดในครั้งเดียว)</li>
      </ul>

      <h2 className="text-xl font-semibold mt-6">🧠 ตัวอย่างการใช้ Redis Caching</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        <code>{`const redis = require("redis");
const express = require("express");
const app = express();
const client = redis.createClient();

app.get("/data", async (req, res) => {
  client.get("data", async (err, cachedData) => {
    if (cachedData) {
      return res.json(JSON.parse(cachedData)); // ส่งข้อมูลจาก cache
    }

    const data = await fetchDataFromDB(); // จำลองดึงจาก DB
    client.setex("data", 3600, JSON.stringify(data)); // cache 1 ชั่วโมง
    res.json(data);
  });
});`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">📉 GZIP Compression</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        <code>{`const compression = require("compression");
app.use(compression());`}</code>
      </pre>

      <p className="mt-4 text-gray-600 dark:text-gray-400">
        ✅ ใช้งานง่าย ช่วยลดขนาดของ response ได้อย่างมีประสิทธิภาพ
      </p>
    </div>
  );
};

export default ApiPerformanceOptimization;
