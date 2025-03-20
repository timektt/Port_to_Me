import React from "react";

const AsyncCallbacks = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold">🔄 Async Callbacks in Node.js</h1>
      
      <p className="mt-4 text-lg">
        ใน Node.js เราใช้ <strong>Callback</strong> เพื่อจัดการกับงานแบบ Asynchronous เช่น การอ่านไฟล์ หรือ Request API
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่างการใช้ Callback</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`const fs = require('fs');

fs.readFile('data.txt', 'utf8', (err, data) => {
  if (err) throw err;
  console.log(data);
});`}</code>
      </pre>

      <p className="mt-4">
        ในโค้ดนี้ เราใช้ <code>fs.readFile()</code> ซึ่งทำงานแบบ <strong>Asynchronous</strong> และรับ Callback เพื่อตอบกลับข้อมูลเมื่ออ่านไฟล์เสร็จ
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 ปัญหา Callback Hell</h2>
      <p className="mt-2">
        เมื่อมีการซ้อน Callback หลายระดับ อาจทำให้โค้ดอ่านยาก เรียกว่า <strong>Callback Hell</strong>
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`fs.readFile('file1.txt', 'utf8', (err, data1) => {
  if (err) throw err;
  fs.readFile('file2.txt', 'utf8', (err, data2) => {
    if (err) throw err;
    fs.readFile('file3.txt', 'utf8', (err, data3) => {
      if (err) throw err;
      console.log(data1, data2, data3);
    });
  });
});`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">🚀 แก้ไข Callback Hell ด้วย Promises</h2>
      <p className="mt-2">
        วิธีแก้ไขปัญหา Callback Hell คือการใช้ <strong>Promises</strong> และ <strong>async/await</strong>
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`const fs = require('fs').promises;

async function readFiles() {
  try {
    const data1 = await fs.readFile('file1.txt', 'utf8');
    const data2 = await fs.readFile('file2.txt', 'utf8');
    const data3 = await fs.readFile('file3.txt', 'utf8');
    console.log(data1, data2, data3);
  } catch (err) {
    console.error(err);
  }
}

readFiles();`}</code>
      </pre>
    </div>
  );
};

export default AsyncCallbacks;