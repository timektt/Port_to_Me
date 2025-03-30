// src/pages/Dashboard.jsx
import React from "react";
import { useAuth } from "../components/context/AuthContext";

const Dashboard = () => {
  const { user } = useAuth();

  return (
    <div className="max-w-3xl mx-auto p-6 space-y-6">
      <h1 className="text-3xl font-bold">📋 User Dashboard</h1>

      <div className="bg-gray-800 p-4 rounded-lg shadow-md">
        <h2 className="text-xl font-semibold mb-2">👤 Profile Info</h2>
        <p><strong>Email:</strong> {user?.email}</p>
        <p><strong>UID:</strong> {user?.uid}</p>
        <p>
          <strong>Email Verified:</strong>{" "}
          {user?.emailVerified ? "✅ Verified" : "❌ Not Verified"}
        </p>
      </div>

      <div className="bg-gray-800 p-4 rounded-lg shadow-md">
        <h2 className="text-xl font-semibold mb-2">⚙️ Account Actions</h2>
        <button
          className="bg-blue-500 hover:bg-blue-600 px-4 py-2 rounded text-white mr-3"
          onClick={() => {
            user.sendEmailVerification();
            alert("Verification email sent.");
          }}
        >
          🔁 Resend Verification Email
        </button>

        <button
          className="bg-green-500 hover:bg-green-600 px-4 py-2 rounded text-white"
          onClick={() => alert("Coming soon: Change password")}
        >
          🔒 Change Password
        </button>
      </div>
    </div>
  );
};

export default Dashboard;
