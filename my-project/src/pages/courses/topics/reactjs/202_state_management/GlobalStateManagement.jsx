import React from "react";

const GlobalStateManagement = () => {
  return (
    <div className="max-w-3xl mx-auto p-6 shadow-lg rounded-lg border">
      <h1 className="text-2xl font-bold">Managing Global State</h1>
      <p className="mt-4">
        <strong>Global State</strong> คือข้อมูลที่ใช้ร่วมกันในหลาย Components และมีหลายแนวทางในการจัดการ เช่น Redux, Context API และ Zustand
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 วิธีจัดการ Global State</h2>
      <ul className="list-disc pl-6 mt-2">
        <li>ใช้ Context API สำหรับข้อมูลที่เปลี่ยนแปลงไม่บ่อย</li>
        <li>ใช้ Redux สำหรับข้อมูลที่ซับซ้อนและต้องจัดการหลาย State</li>
        <li>ใช้ Zustand หรือ Recoil สำหรับข้อมูลที่ต้องการความง่ายและเบา</li>
      </ul>
    </div>
  );
};

export default GlobalStateManagement;
