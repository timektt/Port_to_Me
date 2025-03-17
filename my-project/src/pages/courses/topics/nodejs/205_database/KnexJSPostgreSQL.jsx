import React from "react";

const KnexJSPostgreSQL = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold">⚡ Knex.js & PostgreSQL</h1>
      <p className="mt-4">
        Knex.js เป็น Query Builder ที่รองรับการใช้งานกับ PostgreSQL, MySQL และ SQLite
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 วิธีติดตั้ง Knex.js</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2">
        <code>{`npm install knex pg`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่างการใช้งาน Knex.js</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2">
        <code>{`const knex = require("knex")({
  client: "pg",
  connection: {
    host: "localhost",
    user: "postgres",
    password: "password",
    database: "testdb",
  },
});

async function fetchUsers() {
  const users = await knex("users").select("*");
  console.log("Users:", users);
}

fetchUsers();`}</code>
      </pre>

      <p className="mt-4">🔹 Knex.js ช่วยให้ Query Database ได้ง่ายขึ้น</p>
    </div>
  );
};

export default KnexJSPostgreSQL;
