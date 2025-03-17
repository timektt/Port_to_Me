import React from "react";

const PostgreSQLIntegration = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold">🐘 PostgreSQL Integration with Node.js</h1>
      <p className="mt-4">
        PostgreSQL เป็น Relational Database ที่มีความสามารถสูง รองรับการ Query ขั้นสูง
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 วิธีติดตั้ง PostgreSQL</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2">
        <code>{`npm install pg`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่างการเชื่อมต่อ PostgreSQL</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2">
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
    console.log("Connected to PostgreSQL");
  } catch (error) {
    console.error("Connection failed", error);
  }
}

connectDB();`}</code>
      </pre>

      <p className="mt-4">🚀 ทดสอบโดยรันคำสั่ง **node app.js**</p>
    </div>
  );
};

export default PostgreSQLIntegration;
