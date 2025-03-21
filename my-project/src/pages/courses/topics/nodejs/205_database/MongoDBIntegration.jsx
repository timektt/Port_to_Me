import React from "react";

const MongoDBIntegration = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold">🍃 MongoDB Integration with Node.js</h1>

      <p className="mt-4">
        MongoDB เป็น NoSQL Database ที่ใช้งานง่าย รวดเร็ว และเหมาะกับแอปพลิเคชันที่มีการเปลี่ยนแปลงโครงสร้างข้อมูลบ่อยๆ เช่น แอปโซเชียล หรือระบบจัดการคอนเทนต์
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 การติดตั้ง MongoDB Driver</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`npm install mongodb`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">🔗 การเชื่อมต่อกับ MongoDB</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`const { MongoClient } = require("mongodb");

const uri = "mongodb://localhost:27017";
const client = new MongoClient(uri);

async function connectDB() {
  try {
    await client.connect();
    console.log("✅ Connected to MongoDB");

    const db = client.db("myapp");
    const collection = db.collection("users");

    // เพิ่มข้อมูล
    await collection.insertOne({ name: "Alice", age: 28 });

    // อ่านข้อมูล
    const users = await collection.find({}).toArray();
    console.log("Users:", users);

    // อัปเดตข้อมูล
    await collection.updateOne({ name: "Alice" }, { $set: { age: 30 } });

    // ลบข้อมูล
    await collection.deleteOne({ name: "Alice" });

  } catch (error) {
    console.error("❌ Connection failed", error);
  } finally {
    await client.close();
  }
}

connectDB();`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">📌 คำอธิบายแต่ละคำสั่ง</h2>
      <ul className="list-disc pl-5 mt-2">
        <li><strong>insertOne()</strong> – เพิ่มเอกสารลงใน collection</li>
        <li><strong>find()</strong> – อ่านข้อมูลทั้งหมดจาก collection</li>
        <li><strong>updateOne()</strong> – แก้ไขข้อมูลที่ match เงื่อนไข</li>
        <li><strong>deleteOne()</strong> – ลบเอกสารหนึ่งรายการ</li>
      </ul>

      <p className="mt-6">
        🔎 <strong>Tip:</strong> คุณสามารถใช้ MongoDB Compass เพื่อดูข้อมูลแบบ UI หรือใช้ cloud MongoDB อย่าง Atlas ก็ได้
      </p>
    </div>
  );
};

export default MongoDBIntegration;
