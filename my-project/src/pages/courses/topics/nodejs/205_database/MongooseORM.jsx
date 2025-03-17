import React from "react";

const MongooseORM = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold">🔗 Mongoose ORM for MongoDB</h1>
      <p className="mt-4">
        Mongoose เป็น Object Data Modeling (ODM) สำหรับ MongoDB ทำให้การจัดการข้อมูลง่ายขึ้น
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 วิธีติดตั้ง Mongoose</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2">
        <code>{`npm install mongoose`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่าง Schema & Model</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2">
        <code>{`const mongoose = require("mongoose");

mongoose.connect("mongodb://localhost:27017/mydb");

const userSchema = new mongoose.Schema({
  name: String,
  age: Number,
});

const User = mongoose.model("User", userSchema);

async function createUser() {
  const newUser = new User({ name: "Alice", age: 25 });
  await newUser.save();
  console.log("User Created:", newUser);
}

createUser();`}</code>
      </pre>

      <p className="mt-4">🔧 ใช้ Mongoose แทน MongoDB Driver ได้สะดวกกว่า</p>
    </div>
  );
};

export default MongooseORM;
