import React from "react";

const QueriesMutations = () => {
  return (
    <div className="max-w-4xl mx-auto p-6 text-left">
      <h1 className="text-3xl font-bold mb-4">⚙️ Queries & Mutations</h1>

      <p className="mb-4 text-gray-700 dark:text-gray-300">
        ใน GraphQL มีคำสั่งหลัก 2 ประเภท:
        <br />
        ✅ <strong>Query</strong>: ใช้สำหรับดึงข้อมูลจาก Server<br />
        ✏️ <strong>Mutation</strong>: ใช้สำหรับเปลี่ยนแปลงข้อมูล เช่น สร้าง อัปเดต หรือลบ
      </p>

      <h2 className="text-xl font-semibold mt-6 mb-2">📌 ตัวอย่าง Query</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        <code>{`query {
  user(id: "1") {
    name
    email
  }
}`}</code>
      </pre>
      <p className="mt-2 text-gray-700 dark:text-gray-300">
        🔍 Query นี้จะส่งคำขอไปยัง API เพื่อขอข้อมูลของผู้ใช้ที่มี ID เท่ากับ "1"
      </p>

      <h2 className="text-xl font-semibold mt-6 mb-2">📌 ตัวอย่าง Mutation</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        <code>{`mutation {
  updateUser(id: "1", name: "John Doe") {
    id
    name
    email
  }
}`}</code>
      </pre>
      <p className="mt-2 text-gray-700 dark:text-gray-300">
        ✏️ Mutation นี้ใช้สำหรับอัปเดตชื่อของผู้ใช้ ID = "1" เป็น "John Doe"
      </p>

      <div className="mt-6 p-4 bg-blue-100 dark:bg-blue-900 text-blue-900 dark:text-blue-300 rounded-lg shadow">
        💡 <strong>Tip:</strong> ทุกคำสั่งใน GraphQL ไม่ว่าจะเป็น Query หรือ Mutation จะถูกรวมอยู่ใน <code>POST</code> request เดียวไปยัง endpoint เช่น <code>/graphql</code>
      </div>
    </div>
  );
};

export default QueriesMutations;
