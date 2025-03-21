import React from "react";

const PostgreSQLIntegration = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold">🐘 PostgreSQL Integration with Node.js</h1>

      <p className="mt-4">
        PostgreSQL เป็นระบบฐานข้อมูลแบบ SQL ที่มีความสามารถในการจัดการข้อมูลซับซ้อน รองรับ Transaction และการ Query ขั้นสูง
      </p>

      <h2 className="text-xl font-semibold mt-6">📦 ติดตั้งไลบรารี pg</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`npm install pg`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">🔌 การเชื่อมต่อกับฐานข้อมูล</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`const { Client } = require("pg");

const client = new Client({
  user: "postgres",
  host: "localhost",
  database: "testdb",
  password: "password",
  port: 5432,
});

async function connectDB() {
  try {
    await client.connect();
    console.log("✅ Connected to PostgreSQL");
  } catch (error) {
    console.error("❌ Connection failed", error);
  }
}

connectDB();`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">📌 สร้างตาราง users</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`async function createTable() {
  await client.query(\`
    CREATE TABLE IF NOT EXISTS users (
      id SERIAL PRIMARY KEY,
      name VARCHAR(50),
      email VARCHAR(100)
    )
  \`);
  console.log("📋 Table created");
}

createTable();`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">🟢 Insert ข้อมูล</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`async function insertUser() {
  const res = await client.query(
    "INSERT INTO users (name, email) VALUES ($1, $2) RETURNING *",
    ["Alice", "alice@example.com"]
  );
  console.log("Inserted:", res.rows[0]);
}

insertUser();`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">🔍 ดึงข้อมูลทั้งหมด</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`async function fetchUsers() {
  const res = await client.query("SELECT * FROM users");
  console.log("All Users:", res.rows);
}

fetchUsers();`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">✏️ อัปเดตข้อมูล</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`async function updateUser() {
  const res = await client.query(
    "UPDATE users SET name = $1 WHERE id = $2 RETURNING *",
    ["Bob", 1]
  );
  console.log("Updated:", res.rows[0]);
}

updateUser();`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">🗑️ ลบข้อมูล</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`async function deleteUser() {
  const res = await client.query("DELETE FROM users WHERE id = $1 RETURNING *", [1]);
  console.log("Deleted:", res.rows[0]);
}

deleteUser();`}</code>
      </pre>

      <p className="mt-6">⚙️ คำสั่งเหล่านี้ควรรันทีละฟังก์ชัน หรือเขียนสคริปต์แยกเพื่อควบคุมลำดับการทำงาน</p>
    </div>
  );
};

export default PostgreSQLIntegration;
