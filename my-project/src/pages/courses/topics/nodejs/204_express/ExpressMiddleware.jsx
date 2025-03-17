import React from "react";

const ExpressMiddleware = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold">🛠️ Express.js Middleware</h1>
      <p className="mt-4">
        Middleware เป็นฟังก์ชันที่ทำงานก่อนที่ Request จะถูกส่งไปยัง Route Handler
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่าง Middleware</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2">
        <code>{`const express = require("express");
const app = express();

const logger = (req, res, next) => {
  console.log(\`\${req.method} \${req.url}\`);
  next();
};

app.use(logger);

app.get("/", (req, res) => {
  res.send("Hello, Middleware!");
});

app.listen(3000, () => console.log("Server running on port 3000"));`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">📌 Built-in Middleware</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2">
        <code>{`app.use(express.json());
app.use(express.urlencoded({ extended: true }));`}</code>
      </pre>

      <p className="mt-4">⚡ ใช้สำหรับการรับ JSON และ Form Data</p>
    </div>
  );
};

export default ExpressMiddleware;
