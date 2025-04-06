import React, { useState, useEffect } from "react";
import { sendPasswordResetEmail } from "firebase/auth";
import { auth } from "../firebase/firebase-config";
import { useNavigate } from "react-router-dom";

const ForgotPassword = () => {
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [cooldown, setCooldown] = useState(0);
  const navigate = useNavigate();

  // ✅ ลด spam: Cooldown timer
  useEffect(() => {
    if (cooldown > 0) {
      const timer = setTimeout(() => setCooldown(cooldown - 1), 1000);
      return () => clearTimeout(timer);
    }
  }, [cooldown]);

  const isValidEmail = (email) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  const handleResetPassword = async (e) => {
    e.preventDefault();
    const trimmedEmail = email.trim();

    if (!trimmedEmail) {
      return alert("❌ Please enter your email address.");
    }

    if (!isValidEmail(trimmedEmail)) {
      return alert("❌ Invalid email format.");
    }

    if (cooldown > 0) {
      return alert(`⏳ Please wait ${cooldown} seconds before trying again.`);
    }

    setLoading(true);
    try {
      await sendPasswordResetEmail(auth, trimmedEmail);
      alert("✅ Password reset link sent! Please check your email.");
      setCooldown(30); // ⏱ ป้องกัน spam ภายใน 30 วินาที
      navigate("/login");
    } catch (err) {
      console.error("Reset password error:", err);
      alert("❌ Failed to send reset link: " + err.message);
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
            disabled={loading || cooldown > 0}
            className="w-full bg-green-600 hover:bg-green-700 transition rounded-full py-2 font-bold"
          >
            {loading
              ? "Sending..."
              : cooldown > 0
              ? `Please wait (${cooldown}s)`
              : "Send Reset Link"}
          </button>
        </form>

        <p className="mt-4 text-sm text-purple-300">
          Remember your password?{" "}
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
