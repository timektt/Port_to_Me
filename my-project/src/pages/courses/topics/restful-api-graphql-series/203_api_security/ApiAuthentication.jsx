import React from "react";

const ApiAuthentication = () => {
  return (
    <div className="max-w-4xl mx-auto p-6 text-left">
      <h1 className="text-3xl font-bold mb-4">Authentication & Authorization</h1>
      <p className="mb-4">
        การตรวจสอบตัวตน (Authentication) และ การกำหนดสิทธิ์ (Authorization) เป็นหัวใจสำคัญของ API Security.
      </p>
      <h2 className="text-xl font-semibold mt-4">ตัวอย่าง Basic Authentication</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg">
        {`const express = require("express");
const app = express();

app.use((req, res, next) => {
  const auth = req.headers.authorization;
  if (auth === "Bearer my-secret-token") {
    next();
  } else {
    res.status(401).send("Unauthorized");
  }
});`}
      </pre>
    </div>
  );
};

export default ApiAuthentication;
