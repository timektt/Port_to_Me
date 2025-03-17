import React from "react";

const MiddlewareConcepts = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold">🛠️ Middleware Concepts</h1>
      <p className="mt-4">
        Middleware เป็นฟังก์ชันที่ทำงานก่อนที่ Request จะไปถึง Route Handler เช่น **Logging, Authentication, Error Handling**
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

      <p className="mt-4">⚡ ลองรันแล้วเช็ค console.log()</p>
    </div>
  );
};

export default MiddlewareConcepts;
