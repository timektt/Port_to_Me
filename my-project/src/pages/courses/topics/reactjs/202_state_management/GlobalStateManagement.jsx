import React from "react";

const GlobalStateManagement = () => {
  return (
    <div className="max-w-3xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-4">Managing Global State</h1>

      <p className="mb-4">
        Global State คือข้อมูลที่ถูกแชร์ร่วมกันในหลาย Components เช่น ข้อมูลผู้ใช้ที่เข้าสู่ระบบ, การตั้งค่าธีม, หรือสถานะตะกร้าสินค้า
      </p>

      <h2 className="text-2xl font-semibold mt-6 mb-2">📌 วิธีการจัดการ Global State</h2>
      <ul className="list-disc pl-6 space-y-2">
        <li>
          <strong>Context API:</strong> เหมาะสำหรับข้อมูลที่ไม่เปลี่ยนบ่อย เช่น ธีมหรือภาษา
        </li>
        <li>
          <strong>Redux:</strong> ใช้สำหรับแอปที่มี State หลายตัวและต้องการการจัดการที่ซับซ้อน
        </li>
        <li>
          <strong>Zustand / Recoil:</strong> เป็นทางเลือกใหม่ที่เขียนง่าย ใช้งานเบา และไม่ต้อง boilerplate เยอะ
        </li>
      </ul>

      <h2 className="text-2xl font-semibold mt-6 mb-2">🎯 เลือกใช้แบบไหนดี?</h2>
      <p className="mb-4">
        - ถ้าโปรเจคขนาดเล็กหรือกลาง ใช้ <code className="bg-gray-200 dark:bg-gray-800 px-1 rounded">Context API</code><br />
        - ถ้ามี State หลายตัวและต้องการจัดการ Actions ชัดเจน ใช้ <code className="bg-gray-200 dark:bg-gray-800 px-1 rounded">Redux Toolkit</code><br />
        - ถ้าอยากได้โค้ดที่สั้นและง่าย ลอง <code className="bg-gray-200 dark:bg-gray-800 px-1 rounded">Zustand</code>
      </p>

      <h2 className="text-2xl font-semibold mt-6 mb-2">🧪 ตัวอย่างคำสั่งติดตั้ง Redux</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto text-sm">
        <code>npm install @reduxjs/toolkit react-redux</code>
      </pre>
    </div>
  );
};

export default GlobalStateManagement;
