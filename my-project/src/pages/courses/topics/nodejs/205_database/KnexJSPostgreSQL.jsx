import React from "react";

const KnexJSPostgreSQL = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold">⚡ Knex.js & PostgreSQL</h1>

      <p className="mt-4">
        Knex.js เป็น Query Builder ที่ช่วยให้การทำงานกับฐานข้อมูล SQL อย่าง PostgreSQL, MySQL, SQLite มีความสะดวกมากขึ้น
        สามารถเขียนคำสั่ง SQL ด้วย JavaScript ได้อย่างชัดเจนและปลอดภัย
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 วิธีติดตั้ง Knex.js และ PostgreSQL Driver</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`npm install knex pg`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">🔗 การตั้งค่าเชื่อมต่อฐานข้อมูล</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`const knex = require("knex")({
  client: "pg",
  connection: {
    host: "localhost",
    user: "postgres",
    password: "password",
    database: "testdb",
  },
});`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">📄 SELECT: อ่านข้อมูล</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`async function fetchUsers() {
  const users = await knex("users").select("*");
  console.log("Users:", users);
}

fetchUsers();`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">📝 INSERT: เพิ่มข้อมูล</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`await knex("users").insert({ name: "Alice", email: "alice@example.com" });`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">✏️ UPDATE: แก้ไขข้อมูล</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`await knex("users").where({ id: 1 }).update({ name: "Updated Alice" });`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">🗑️ DELETE: ลบข้อมูล</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`await knex("users").where({ id: 1 }).del();`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">📂 การใช้ Migration</h2>
      <p className="mt-2">Knex มีระบบ migration สำหรับจัดการโครงสร้างตาราง:</p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`npx knex init
npx knex migrate:make create_users_table`}</code>
      </pre>

      <p className="mt-4">🔧 ตัวอย่างไฟล์ migration:</p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`exports.up = function(knex) {
  return knex.schema.createTable("users", function(table) {
    table.increments("id");
    table.string("name");
    table.string("email");
  });
};

exports.down = function(knex) {
  return knex.schema.dropTable("users");
};`}</code>
      </pre>

      <p className="mt-6">
        ✅ <strong>สรุป:</strong> Knex.js เป็นเครื่องมือที่ทำให้การทำงานกับ SQL มีประสิทธิภาพ ใช้งานง่าย และเหมาะกับโปรเจกต์ Node.js ทุกระดับ
      </p>
    </div>
  );
};

export default KnexJSPostgreSQL;
