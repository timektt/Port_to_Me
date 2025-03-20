import React from "react";

const AsyncErrors = () => {
  return (
    <div className="max-w-3xl mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">⚠️ Handling Asynchronous Errors</h1>
      <p>
        ใน Node.js เราต้องจัดการข้อผิดพลาดในโค้ดที่ทำงานแบบ Asynchronous ให้เหมาะสม เช่น การใช้ <code>try-catch</code> และ <code>event emitters</code>
      </p>
      
      <h2 className="text-xl font-semibold mt-6">📌 ใช้ try-catch กับ async/await</h2>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        <pre>
          <code>{`const fs = require('fs').promises;

async function readFile() {
  try {
    const data = await fs.readFile('nonexistent.txt', 'utf8');
    console.log(data);
  } catch (err) {
    console.error('เกิดข้อผิดพลาดในการอ่านไฟล์:', err.message);
  }
}

readFile();`}</code>
        </pre>
      </div>
      
      <h2 className="text-xl font-semibold mt-6">📌 ใช้ EventEmitter จัดการข้อผิดพลาด</h2>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        <pre>
          <code>{`const EventEmitter = require('events');
const myEmitter = new EventEmitter();

myEmitter.on('error', (err) => {
  console.error('มีข้อผิดพลาดเกิดขึ้น:', err.message);
});

myEmitter.emit('error', new Error('Something went wrong!'));`}</code>
        </pre>
      </div>
      
      <h2 className="text-xl font-semibold mt-6">📌 สรุป</h2>
      <p>
        ✅ การจัดการข้อผิดพลาดใน Node.js มีหลายวิธี เช่น <code>try-catch</code> สำหรับ async/await และ <code>EventEmitter</code> สำหรับเหตุการณ์ที่เกิดขึ้นในระบบ
      </p>
    </div>
  );
};

export default AsyncErrors;
