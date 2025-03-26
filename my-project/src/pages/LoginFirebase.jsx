import React from "react";
import { auth } from "../firebase/firebase-config";
import { GoogleAuthProvider, GithubAuthProvider, signInWithPopup } from "firebase/auth";
import { useNavigate } from "react-router-dom";

const LoginFirebase = () => {
  const navigate = useNavigate();

  const handleLogin = async (providerType) => {
    const provider =
      providerType === "google" ? new GoogleAuthProvider() : new GithubAuthProvider();

    try {
      const result = await signInWithPopup(auth, provider);
      const user = result.user;

      // Save user info or token if needed
      localStorage.setItem("token", await user.getIdToken());
      alert("Login Success via " + providerType);
      navigate("/");
    } catch (err) {
      console.error("Login Error:", err);
      alert("Login failed");
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-900 to-black text-white px-4">
      <div className="bg-gray-800 p-6 rounded-2xl shadow-lg w-full max-w-sm text-center">
        <h1 className="text-2xl font-bold mb-4">Superbear Firebase Login</h1>
        <button
          onClick={() => handleLogin("google")}
          className="w-full mb-3 bg-red-500 hover:bg-red-600 py-2 px-4 rounded-full font-semibold"
        >
          Sign in with Google
        </button>
        <button
          onClick={() => handleLogin("github")}
          className="w-full bg-gray-700 hover:bg-gray-600 py-2 px-4 rounded-full font-semibold"
        >
          Sign in with GitHub
        </button>
      </div>
    </div>
  );
};

export default LoginFirebase;
