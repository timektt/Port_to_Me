import React from "react";

const ApiTestingMonitoring = () => {
  return (
    <div className="max-w-4xl mx-auto p-6 text-left">
      <h1 className="text-3xl font-bold mb-4">🧪 Testing & 📈 Monitoring APIs</h1>
      <p className="mb-4">
        การทดสอบและติดตาม API เป็นกระบวนการที่ช่วยให้มั่นใจว่า API ทำงานได้ถูกต้อง และมีประสิทธิภาพที่ดีเมื่อใช้งานจริง
      </p>

      <h2 className="text-xl font-semibold mt-6">✅ การทดสอบ API ด้วย Jest + Supertest</h2>
      <p className="mt-2">เหมาะสำหรับทดสอบ Endpoint ว่าทำงานตามที่คาดไว้หรือไม่</p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        <code>{`const request = require("supertest");
const app = require("../app");

test("GET /users", async () => {
  const response = await request(app).get("/users");
  expect(response.status).toBe(200);
});`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">📊 การ Monitoring API</h2>
      <p className="mt-2">
        สามารถใช้เครื่องมือต่างๆ เพื่อติดตามสุขภาพ API เช่น:
      </p>
      <ul className="list-disc ml-6 mt-2 space-y-1">
        <li>🔍 <strong>Prometheus</strong> สำหรับเก็บ Metrics</li>
        <li>📈 <strong>Grafana</strong> สำหรับ Visualization</li>
        <li>📬 <strong>Log Monitoring</strong> เช่น ELK Stack (Elasticsearch + Logstash + Kibana)</li>
        <li>⏱️ <strong>Uptime Robot / Pingdom</strong> สำหรับตรวจสอบการออนไลน์ของ API</li>
      </ul>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่าง Health Check API</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        <code>{`app.get("/health", (req, res) => {
  res.status(200).json({ status: "ok", uptime: process.uptime() });
});`}</code>
      </pre>
      <p className="mt-2 text-gray-600 dark:text-gray-400">
        ใช้ร่วมกับเครื่องมือตรวจสอบ Uptime ได้อย่างมีประสิทธิภาพ
      </p>
    </div>
  );
};

export default ApiTestingMonitoring;
