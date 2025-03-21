import React from "react";

const RestCrud = () => {
  return (
    <div className="max-w-3xl mx-auto p-6 text-left">
      <h1 className="text-3xl font-bold mb-4">CRUD ใน REST API</h1>
      <p className="mb-4">
        CRUD ย่อมาจาก <strong>Create, Read, Update, Delete</strong> 
        ซึ่งเป็นพื้นฐานของการทำงานกับข้อมูลใน REST API
      </p>

      <h2 className="text-2xl font-semibold mt-6 mb-2">📌 การทำงานของ CRUD</h2>
      <ul className="list-disc pl-6 space-y-2">
        <li><strong>Create (POST):</strong> เพิ่มข้อมูลใหม่</li>
        <li><strong>Read (GET):</strong> อ่านข้อมูล</li>
        <li><strong>Update (PUT/PATCH):</strong> แก้ไขข้อมูล</li>
        <li><strong>Delete (DELETE):</strong> ลบข้อมูล</li>
      </ul>

      <h2 className="text-2xl font-semibold mt-6 mb-2">📌 ตัวอย่างการทำ CRUD ด้วย Express.js</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto">
        <code>
{`app.post("/api/users", (req, res) => {
  // Create User
});

app.get("/api/users", (req, res) => {
  // Get Users
});

app.put("/api/users/:id", (req, res) => {
  // Update User
});

app.delete("/api/users/:id", (req, res) => {
  // Delete User
});`}
        </code>
      </pre>
    </div>
  );
};

export default RestCrud;
