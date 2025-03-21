import React from "react";

const ApiAuthentication = () => {
  return (
    <div className="max-w-4xl mx-auto p-6 text-left">
      <h1 className="text-3xl font-bold mb-4">🔐 Authentication & Authorization</h1>

      <p className="mb-4 text-gray-700 dark:text-gray-300">
        <strong>Authentication</strong> คือ การตรวจสอบว่า "คุณคือใคร"<br />
        <strong>Authorization</strong> คือ การกำหนดว่า "คุณสามารถทำอะไรได้บ้าง"
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่าง Basic Authentication</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        <code>{`const express = require("express");
const app = express();

app.use((req, res, next) => {
  const auth = req.headers.authorization;
  if (auth === "Bearer my-secret-token") {
    next();
  } else {
    res.status(401).send("Unauthorized");
  }
});`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่างการใช้ JWT Authentication</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        <code>{`const jwt = require("jsonwebtoken");

app.use((req, res, next) => {
  const token = req.headers.authorization?.split(" ")[1];
  if (!token) return res.status(401).json({ message: "Token required" });

  try {
    const decoded = jwt.verify(token, "my-secret-key");
    req.user = decoded;
    next();
  } catch (err) {
    res.status(403).json({ message: "Invalid token" });
  }
});`}</code>
      </pre>

      <div className="mt-6 p-4 bg-yellow-100 dark:bg-yellow-900 text-yellow-800 dark:text-yellow-200 rounded-lg shadow">
        ⚠️ <strong>Security Tip:</strong> อย่าเก็บ token หรือ key ไว้ในโค้ดโดยตรง ควรเก็บใน environment variables (.env)
      </div>
    </div>
  );
};

export default ApiAuthentication;
