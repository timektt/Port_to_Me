import React from "react";

const MongooseORM = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold">🔗 Mongoose ORM for MongoDB</h1>

      <p className="mt-4">
        Mongoose เป็นไลบรารีที่ช่วยให้สามารถจัดการ MongoDB ด้วยรูปแบบ Object ได้ง่ายและปลอดภัยยิ่งขึ้น
        โดยใช้แนวคิด Schema และ Model ในการควบคุมโครงสร้างข้อมูล
      </p>

      <h2 className="text-xl font-semibold mt-6">📦 ติดตั้ง Mongoose</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`npm install mongoose`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">🧱 สร้าง Schema และ Model</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`const mongoose = require("mongoose");

mongoose.connect("mongodb://localhost:27017/mydb", {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});

const userSchema = new mongoose.Schema({
  name: { type: String, required: true },
  age: Number,
  email: String,
  createdAt: { type: Date, default: Date.now }
});

const User = mongoose.model("User", userSchema);`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">✅ สร้างผู้ใช้ใหม่</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`async function createUser() {
  const newUser = new User({ name: "Alice", age: 25, email: "alice@example.com" });
  await newUser.save();
  console.log("User Created:", newUser);
}

createUser();`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">🔍 ค้นหาผู้ใช้</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`async function findUsers() {
  const users = await User.find();
  console.log("All Users:", users);
}

findUsers();`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">✏️ อัปเดตข้อมูล</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`async function updateUser() {
  const result = await User.updateOne({ name: "Alice" }, { age: 30 });
  console.log("Update Result:", result);
}

updateUser();`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">🗑️ ลบผู้ใช้</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`async function deleteUser() {
  const result = await User.deleteOne({ name: "Alice" });
  console.log("Delete Result:", result);
}

deleteUser();`}</code>
      </pre>

      <p className="mt-6">
        💡 <strong>สรุป:</strong> Mongoose ช่วยให้เราจัดการ MongoDB ได้ง่ายขึ้น ด้วย Schema ที่กำหนดชัดเจน และรองรับ validation, middleware, population และอื่น ๆ อีกมาก
      </p>
    </div>
  );
};

export default MongooseORM;
