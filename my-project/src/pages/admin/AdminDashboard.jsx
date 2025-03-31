// src/pages/admin/AdminDashboard.jsx
import React from "react";

const AdminDashboard = () => {
  console.log("✅ AdminDashboard rendered");

  return (
    <div className="p-10">
      <h1 className="text-4xl font-bold py-10 text-yellow-400 mb-4">
        🛡️ Welcome, Admin
      </h1>
      <p className="text-lg text-white">You have full access to manage the system.</p>
      {/* 🔧 Future: เพิ่มปุ่ม จัดการคอร์ส/ผู้ใช้ได้ที่นี่ */}
    </div>
  );
};

export default AdminDashboard;
