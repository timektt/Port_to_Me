import React, { useState } from "react";
import { auth } from "../firebase/firebase-config";
import {
  signInWithEmailAndPassword,
  GoogleAuthProvider,
  GithubAuthProvider,
  signInWithPopup,
  linkWithCredential,
  fetchSignInMethodsForEmail,
  sendPasswordResetEmail,
} from "firebase/auth";
import { useNavigate } from "react-router-dom";
import { FcGoogle } from "react-icons/fc";
import { FaGithub } from "react-icons/fa";
import { AiFillEye, AiFillEyeInvisible } from "react-icons/ai";
import { useLocation } from "react-router-dom";
const Login = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
const from = location.state?.from?.pathname || "/";

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const result = await signInWithEmailAndPassword(auth, email, password);
      const user = result.user;

      if (!user.emailVerified) {
        await user.sendEmailVerification();
        alert("Please verify your email. A verification link has been sent.");
        setLoading(false);
        return;
      }

      const token = await user.getIdToken(true);
      localStorage.setItem("token", token);
      alert("Login Success!");
      navigate(from, { replace: true });
    } catch (err) {
      console.error("Login error:", err);
      alert("Login failed: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleOAuthLogin = async (provider) => {
    try {
      const result = await signInWithPopup(auth, provider);
      const user = result.user;
      localStorage.setItem("token", await user.getIdToken());
      alert(`Welcome, ${user.displayName}`);
      navigate("/");
    } catch (err) {
      if (err.code === "auth/account-exists-with-different-credential") {
        const email = err.customData?.email;
        const pendingCred = err.credential;
        const methods = await fetchSignInMethodsForEmail(auth, email);

        if (methods.includes("google.com")) {
          alert(
            `บัญชีนี้เคยเชื่อมต่อผ่าน Google มาก่อน กรุณาใช้ปุ่ม "Sign in with Google"`
          );
          try {
            const googleResult = await signInWithPopup(auth, new GoogleAuthProvider());
            await linkWithCredential(googleResult.user, pendingCred);
            localStorage.setItem("token", await googleResult.user.getIdToken());
            alert("บัญชี GitHub เชื่อมกับบัญชี Google สำเร็จ!");
            navigate("/");
          } catch (linkErr) {
            console.error("Error linking credentials:", linkErr);
            alert("เกิดข้อผิดพลาดในการเชื่อมบัญชี");
          }
        } else {
          alert("บัญชีนี้เคยลงทะเบียนด้วยผู้ให้บริการอื่น กรุณาเข้าสู่ระบบด้วยวิธีเดิม");
        }
      } else {
        console.error(err);
        alert("OAuth login failed");
      }
    }
  };

  const handleForgotPassword = () => {
    navigate("/forgot-password");
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-black to-gray-800 text-white px-4">
      <div className="bg-gradient-to-br from-gray-900 to-gray-800 p-6 rounded-2xl shadow-lg w-full max-w-sm text-center">
        <h1 className="text-2xl font-bold mb-1">Superbear</h1>
        <p className="text-gray-300 mb-5">Sign in to your account</p>

        <form onSubmit={handleLogin} className="space-y-3">
          <input
            type="text"
            placeholder="Email"
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
              aria-label={showPassword ? "Hide password" : "Show password"}
            >
              {showPassword ? <AiFillEyeInvisible /> : <AiFillEye />}
            </button>
          </div>

          <div className="text-left text-sm text-purple-300 mb-2 ml-1">
            <span
              onClick={handleForgotPassword}
              className="hover:underline cursor-pointer"
            >
              Forgot Password?
            </span>
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
          onClick={() => handleOAuthLogin(new GoogleAuthProvider())}
          className="bg-gray-700 hover:bg-gray-600 w-full flex items-center justify-center gap-3 rounded-lg py-2 font-semibold transition"
        >
          <FcGoogle className="text-xl" />
          Sign in with Google
        </button>

        <button
          onClick={() => handleOAuthLogin(new GithubAuthProvider())}
          className="bg-gray-800 hover:bg-gray-700 w-full flex items-center justify-center gap-3 rounded-lg py-2 font-semibold transition mt-2"
        >
          <FaGithub className="text-xl" />
          Sign in with GitHub
        </button>

        <p className="mt-4 text-sm text-gray-400">
          Don't have an account?{" "}
          <span
            className="text-blue-400 hover:underline cursor-pointer"
            onClick={() => navigate("/register")}
          >
            Register here
          </span>
        </p>
      </div>
    </div>
  );
};

export default Login;
