import React from "react";

const ExpressRouting = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold">📍 Express.js Routing</h1>
      <p className="mt-4">
        Routing ใน Express ใช้สำหรับกำหนดเส้นทางในการเข้าถึง API
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่างการกำหนด Route</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2">
        <code>{`const express = require("express");
const app = express();

app.get("/", (req, res) => res.send("Home Page"));
app.get("/about", (req, res) => res.send("About Page"));
app.get("/contact", (req, res) => res.send("Contact Page"));

app.listen(3000, () => console.log("Server running on port 3000"));`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">📌 Dynamic Route</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2">
        <code>{`app.get("/user/:id", (req, res) => {
  res.send(\`User ID: \${req.params.id}\`);
});`}</code>
      </pre>

      <p className="mt-4">🚀 ทดสอบโดยเปิด **http://localhost:3000/user/123**</p>
    </div>
  );
};

export default ExpressRouting;
