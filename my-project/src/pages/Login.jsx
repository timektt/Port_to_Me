import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";

const Login = () => {
  const [email, setEmail] = useState("test@example.com");
  const [password, setPassword] = useState("123456");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate(); // ✅ ใช้สำหรับ redirect

  const handleLogin = async (e) => {
    e.preventDefault();
    if (loading) return;
    setLoading(true);

    try {
      const res = await axios.post(`${import.meta.env.VITE_API_URL}/api/login`, {
        email,
        password,
      });

      if (res.data.success) {
        localStorage.setItem("token", res.data.token); // ✅ บันทึก token
        navigate("/"); // ✅ redirect ไปหน้าแรก
      } else {
        alert("Invalid credentials");
      }
    } catch (err) {
      console.error(err);
      alert("Login failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-black to-gray-800 text-white px-4">
      <div className="bg-gradient-to-br from-gray-900 to-gray-800 p-6 rounded-2xl shadow-lg w-full max-w-sm text-center">
        <h1 className="text-2xl font-bold mb-1">Superbear</h1>
        <p className="text-gray-300 mb-5">Sign in to your account</p>

        <form onSubmit={handleLogin} className="space-y-3">
          <input
            type="text"
            placeholder="User ID"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full px-4 py-2 rounded-xl bg-gray-600 text-white placeholder-gray-300 focus:outline-none"
            required
          />

          <div className="relative">
            <input
              type={showPassword ? "text" : "password"}
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full px-4 py-2 rounded-xl bg-gray-600 text-white placeholder-gray-300 focus:outline-none"
              required
            />
            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              className="absolute right-3 top-2 text-lg text-gray-300"
            >
              {showPassword ? "🙈" : "👁️"}
            </button>
          </div>

          <div className="text-left text-sm text-purple-300 mb-2 ml-1">
            <a
              href="#"
              onClick={(e) => {
                e.preventDefault();
                alert("Please contact support to reset your password.");
              }}
              className="hover:underline"
            >
              Forgot Password
            </a>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-purple-600 hover:bg-purple-700 transition rounded-full py-2 font-bold"
          >
            {loading ? "Logging in..." : "Login"}
          </button>
        </form>

        <p className="my-3 text-gray-400 font-semibold">OR</p>

        <button
          onClick={() =>
            window.location.href = "https://github.com/login/oauth/authorize"
          }
          className="bg-gray-700 hover:bg-gray-600 w-full flex items-center justify-center gap-3 rounded-lg py-2 font-semibold transition"
        >
          <img
            src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg"
            alt="GitHub"
            className="w-5 h-5"
          />
          Sign in with GitHub
        </button>
      </div>
    </div>
  );
};

export default Login;
