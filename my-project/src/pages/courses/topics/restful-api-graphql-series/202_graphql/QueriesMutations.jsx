import React from "react";

const QueriesMutations = () => {
  return (
    <div className="max-w-4xl mx-auto p-6 text-left">
      <h1 className="text-3xl font-bold mb-4">Queries & Mutations</h1>
      <p className="mb-4">
        GraphQL มีสองประเภทหลักในการสื่อสารกับ API: <strong>Queries</strong> (ใช้ดึงข้อมูล) และ <strong>Mutations</strong> (ใช้เปลี่ยนแปลงข้อมูล)
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg">
        {`query {
  user(id: "1") {
    name
    email
  }
}`}
      </pre>
      <p className="mt-4">Mutation ใช้สำหรับสร้าง, อัปเดต หรือ ลบข้อมูล</p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg">
        {`mutation {
  updateUser(id: "1", name: "John Doe") {
    name
    email
  }
}`}
      </pre>
    </div>
  );
};

export default QueriesMutations;
