import React from "react";

const ErrorHandling = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold">⚠️ Error Handling in Express</h1>
      <p className="mt-4">
        การจัดการข้อผิดพลาด (<strong>Error Handling</strong>) ช่วยให้ API ของเราทำงานได้อย่างถูกต้องและปลอดภัยมากขึ้น
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่าง Error Handling Middleware</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`const express = require("express");
const app = express();

app.get("/error", (req, res, next) => {
  next(new Error("Something went wrong!"));
});

// Middleware จัดการ Error
app.use((err, req, res, next) => {
  res.status(500).json({ message: err.message });
});

app.listen(3000, () => console.log("Server running on port 3000"));`}</code>
      </pre>

      <p className="mt-4">🔥 ทดสอบโดยเรียก <strong>http://localhost:3000/error</strong></p>

      <h2 className="text-xl font-semibold mt-6">✅ ข้อแนะนำเพิ่มเติม</h2>
      <ul className="list-disc ml-5 mt-2 space-y-2">
        <li>ให้แยก Error Middleware ไปไว้ในไฟล์แยกเพื่อให้โค้ดสะอาด</li>
        <li>สร้างคลาส Error แบบกำหนดเอง เช่น NotFoundError, ValidationError</li>
        <li>อย่าลืมใช้ try-catch ใน async/await ร่วมกับ next()</li>
      </ul>

      <h2 className="text-xl font-semibold mt-6">📘 ตัวอย่างการใช้ try-catch ใน async/await</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`app.get("/user", async (req, res, next) => {
  try {
    const user = await getUserFromDb();
    res.json(user);
  } catch (err) {
    next(err);
  }
});`}</code>
      </pre>
    </div>
  );
};

export default ErrorHandling;
