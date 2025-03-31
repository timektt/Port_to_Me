import React from "react";
import { useAuth } from "../components/context/AuthContext";

const Dashboard = () => {
  const { user, role } = useAuth(); // ✅ เพิ่ม role

  return (
    <div className="max-w-5xl mx-auto px-6 py-20 sm:px-6 lg:px-8 space-y-10">
      <h1 className="text-3xl sm:text-4xl font-bold text-center">
         Your Dashboard
      </h1>

      {/* Profile Info Card */}
      <div className="bg-black/90 rounded-2xl p-6 sm:p-8 border border-yellow-500 shadow-xl">
        <h2 className="text-xl sm:text-2xl font-semibold text-yellow-300 mb-4">
          👤 Profile Information
        </h2>
        <div className="space-y-2 text-white text-sm sm:text-base">
          <p>
            <span className="font-semibold text-yellow-400">Email:</span>{" "}
            {user?.email}
          </p>
          <p>
            <span className="font-semibold text-yellow-400">UID:</span>{" "}
            {user?.uid}
          </p>
          <p>
            <span className="font-semibold text-yellow-400">Email Verified:</span>{" "}
            {user?.emailVerified ? "✅ Verified" : "❌ Not Verified"}
          </p>
          <p>
            <span className="font-semibold text-yellow-400">Role:</span>{" "}
            {role ? `👑 ${role}` : "—"}
          </p>
        </div>
      </div>

      {/* Account Actions */}
      <div className="bg-black/90 rounded-2xl p-6 sm:p-8 border border-yellow-500 shadow-xl">
        <h2 className="text-xl sm:text-2xl font-semibold text-yellow-300 mb-4">
          ⚙️ Account Actions
        </h2>
        <div className="flex flex-col sm:flex-row sm:items-center gap-4">
          <button
            className="w-full sm:w-auto bg-gradient-to-r from-yellow-500 to-yellow-400 text-black font-bold px-6 py-2 rounded-full shadow-md hover:shadow-yellow-500/50 transition"
            onClick={() => {
              user?.sendEmailVerification();
              alert("Verification email sent.");
            }}
          >
            🔁 Resend Verification Email
          </button>
          <button
            className="w-full sm:w-auto bg-gradient-to-r from-yellow-500 to-yellow-400 text-black font-bold px-6 py-2 rounded-full shadow-md hover:shadow-yellow-500/50 transition"
            onClick={() => alert("Coming soon: Change password")}
          >
            🔒 Change Password
          </button>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
