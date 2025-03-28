import React, { useState } from "react";
import { sendPasswordResetEmail } from "firebase/auth";
import { auth } from "../firebase/firebase-config";
import { useNavigate } from "react-router-dom";

const ForgotPassword = () => {
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleResetPassword = async (e) => {
    e.preventDefault();
    if (!email) return alert("Please enter your email address.");
    setLoading(true);
    try {
      await sendPasswordResetEmail(auth, email);
      alert("Password reset link sent! Please check your email.");
      navigate("/login");
    } catch (err) {
      console.error("Reset password error:", err);
      alert("Failed to send reset link: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-black to-gray-800 text-white px-4">
      <div className="bg-gradient-to-br from-gray-900 to-gray-800 p-6 rounded-2xl shadow-lg w-full max-w-sm text-center">
        <h1 className="text-2xl font-bold mb-1">Forgot Password</h1>
        <p className="text-gray-300 mb-5">We'll send you a reset link</p>

        <form onSubmit={handleResetPassword} className="space-y-3">
          <input
            type="email"
            placeholder="Enter your email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full px-4 py-2 rounded-xl bg-gray-600 text-white placeholder-gray-300 focus:outline-none"
            required
          />

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-green-600 hover:bg-green-700 transition rounded-full py-2 font-bold"
          >
            {loading ? "Sending..." : "Send Reset Link"}
          </button>
        </form>

        <p className="mt-4 text-sm text-purple-300">
          Remember your password? {" "}
          <span
            onClick={() => navigate("/login")}
            className="text-blue-400 hover:underline cursor-pointer"
          >
            Login
          </span>
        </p>
      </div>
    </div>
  );
};

export default ForgotPassword;
