// ✅ src/pages/admin/AdminDashboard.jsx
import React from "react";
import AdminUpdates from "./AdminUpdates";

const AdminDashboard = () => {
  console.log("✅ AdminDashboard rendered");

  return (
    <div className="min-h-screen px-4 pt-24 sm:px-6 lg:px-10 pb-20 flex flex-col items-center">
      {/* ✅ Centered Header */}
      <div className="text-center mb-12">
        <h1 className="text-4xl sm:text-5xl font-semibold tracking-tight">
          Welcome, Admin
        </h1>
        <p className="mt-3 text-base text-gray-400">
          You have full access to manage and update the platform content.
        </p>
      </div>

      {/* ✅ Content Section */}
      <div className="w-full max-w-6xl">
        <AdminUpdates />
      </div>
    </div>
  );
};

export default AdminDashboard;
