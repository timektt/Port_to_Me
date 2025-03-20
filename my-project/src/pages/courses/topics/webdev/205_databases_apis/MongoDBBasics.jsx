import React from "react";

const MongoDBBasics = () => {
  return (
    <>
      <h1 className="text-2xl font-bold mb-4">พื้นฐาน MongoDB</h1>
      <p>
        MongoDB เป็นฐานข้อมูลแบบ NoSQL ที่เก็บข้อมูลในรูปแบบ JSON-like documents ซึ่งช่วยให้การพัฒนาเว็บแอปพลิเคชันมีความยืดหยุ่นมากขึ้น
      </p>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">คุณสมบัติของ MongoDB</h2>
      <ul className="list-disc pl-6 mb-4">
        <li>ใช้โครงสร้างแบบ Document-Oriented</li>
        <li>รองรับการขยายขนาดแบบ Horizontal Scaling</li>
        <li>สามารถเก็บข้อมูลแบบ Dynamic Schema ได้</li>
      </ul>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: การเชื่อมต่อ MongoDB ด้วย Node.js</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`const mongoose = require('mongoose');

mongoose.connect('mongodb://localhost:27017/mydatabase', {
  useNewUrlParser: true,
  useUnifiedTopology: true
});

const db = mongoose.connection;
db.on('error', console.error.bind(console, 'connection error:'));
db.once('open', () => {
  console.log('Connected to MongoDB');
});`}
      </pre>
    </>
  );
};

export default MongoDBBasics;
