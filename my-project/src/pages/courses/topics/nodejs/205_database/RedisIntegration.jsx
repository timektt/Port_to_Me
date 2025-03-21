import React from "react";

const RedisIntegration = () => {
  return (
    <div className="max-w-3xl mx-auto p-4">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold mb-4">🚀 Redis Integration in Node.js</h1>

      <p className="mb-4">
        Redis เป็น in-memory data store ที่รวดเร็ว เหมาะสำหรับใช้เป็น cache, session storage, และ pub/sub system
      </p>

      <h2 className="text-xl font-semibold mt-6">📦 ติดตั้ง Redis Client</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto">
        <code>{`npm install redis`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">🔌 การเชื่อมต่อ Redis</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto">
        <code>{`const redis = require("redis");

const client = redis.createClient();

client.connect()
  .then(() => console.log("✅ Connected to Redis"))
  .catch(err => console.error("❌ Redis connection failed", err));`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">📌 Set และ Get ข้อมูล</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto">
        <code>{`async function redisExample() {
  await client.set("username", "Superbear");
  const value = await client.get("username");
  console.log("Username:", value);
}

redisExample();`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">🧠 ใช้ Redis ร่วมกับ Express</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto">
        <code>{`const express = require("express");
const app = express();

app.get("/cache", async (req, res) => {
  const cached = await client.get("hello");
  if (cached) {
    return res.send("🧠 Cache: " + cached);
  }

  // จำลองข้อมูลจากแหล่งช้า
  const data = "Hello from server";
  await client.setEx("hello", 60, data); // cache 60 วินาที
  res.send("🧾 Fresh: " + data);
});

app.listen(3000, () => console.log("Server running on port 3000"));`}</code>
      </pre>

      <p className="mt-6">
        ✅ Redis ช่วยให้แอปสามารถจัดการข้อมูลชั่วคราวและเพิ่มความเร็วในการตอบสนองได้อย่างมีประสิทธิภาพ
      </p>
    </div>
  );
};

export default RedisIntegration;
