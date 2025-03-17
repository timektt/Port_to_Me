import React from "react";

const MongoDBIntegration = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold">🍃 MongoDB Integration with Node.js</h1>
      <p className="mt-4">
        MongoDB เป็น NoSQL Database ที่ใช้งานง่าย รวดเร็ว และเหมาะกับแอปพลิเคชันที่มีการเปลี่ยนแปลงโครงสร้างข้อมูลบ่อยๆ
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 วิธีติดตั้ง MongoDB</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2">
        <code>{`npm install mongodb`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่างการเชื่อมต่อ MongoDB</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2">
        <code>{`const { MongoClient } = require("mongodb");

const uri = "mongodb://localhost:27017";
const client = new MongoClient(uri);

async function connectDB() {
  try {
    await client.connect();
    console.log("Connected to MongoDB");
  } catch (error) {
    console.error("Connection failed", error);
  }
}

connectDB();`}</code>
      </pre>

      <p className="mt-4">🔥 ทดสอบโดยรันคำสั่ง **node app.js**</p>
    </div>
  );
};

export default MongoDBIntegration;
