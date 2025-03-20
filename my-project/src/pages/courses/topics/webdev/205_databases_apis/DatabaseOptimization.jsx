import React from "react";

const DatabaseOptimization = () => {
  return (
    <>
      <h1 className="text-2xl font-bold mb-4">การเพิ่มประสิทธิภาพฐานข้อมูล (Database Optimization & Indexing)</h1>
      <p>
        การเพิ่มประสิทธิภาพของฐานข้อมูลช่วยให้ Query ทำงานได้เร็วขึ้น ลดภาระเซิร์ฟเวอร์ และเพิ่มประสิทธิภาพของแอปพลิเคชัน
      </p>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">เทคนิคการเพิ่มประสิทธิภาพฐานข้อมูล</h2>
      <ul className="list-disc pl-6 mb-4">
        <li>ใช้ Indexes เพื่อลดเวลาในการค้นหาข้อมูล</li>
        <li>ออกแบบโครงสร้างฐานข้อมูลให้มีประสิทธิภาพ (Normalization & Denormalization)</li>
        <li>ใช้ Query Optimization เพื่อลดภาระของ Database Engine</li>
      </ul>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: การสร้าง Index ใน SQL</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`CREATE INDEX idx_user_name ON users(name);`}
      </pre>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: การวิเคราะห์ Query ด้วย EXPLAIN</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`EXPLAIN SELECT * FROM users WHERE name = 'Alice';`}
      </pre>
    </>
  );
};

export default DatabaseOptimization;
