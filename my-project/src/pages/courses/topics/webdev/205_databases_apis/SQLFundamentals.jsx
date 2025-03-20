import React from "react";

const SQLFundamentals = () => {
  return (
    <>
      <h1 className="text-2xl font-bold mb-4">พื้นฐาน SQL</h1>
      <p>
        SQL (Structured Query Language) เป็นภาษาที่ใช้ในการจัดการฐานข้อมูลแบบเชิงสัมพันธ์ (Relational Databases) เช่น MySQL, PostgreSQL, และ SQLite
      </p>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">คำสั่งพื้นฐานของ SQL</h2>
      <ul className="list-disc pl-6 mb-4">
        <li>SELECT - ใช้สำหรับดึงข้อมูลจากฐานข้อมูล</li>
        <li>INSERT - ใช้สำหรับเพิ่มข้อมูลใหม่</li>
        <li>UPDATE - ใช้สำหรับแก้ไขข้อมูลที่มีอยู่</li>
        <li>DELETE - ใช้สำหรับลบข้อมูล</li>
      </ul>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: การใช้คำสั่ง SELECT</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`SELECT * FROM users WHERE age > 18;`}
      </pre>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: การสร้างตารางใหม่</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`CREATE TABLE users (
  id INT PRIMARY KEY,
  name VARCHAR(50),
  age INT
);`}
      </pre>
    </>
  );
};

export default SQLFundamentals;
