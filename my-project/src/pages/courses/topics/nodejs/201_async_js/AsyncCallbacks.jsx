import React from "react";

const AsyncCallbacks = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold">🔄 Async Callbacks in Node.js</h1>
      
      <p className="mt-4 text-lg">
        ใน Node.js เราใช้ <strong>Callback</strong> เพื่อจัดการกับงานแบบ Asynchronous เช่น การอ่านไฟล์ หรือ Request API
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่างการใช้ Callback</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2">
        <code>{`const fs = require('fs');

fs.readFile('data.txt', 'utf8', (err, data) => {
  if (err) throw err;
  console.log(data);
});`}</code>
      </pre>

      <p className="mt-4">
        ในโค้ดนี้ เราใช้ `fs.readFile()` ซึ่งทำงานแบบ **Asynchronous** และรับ Callback เพื่อตอบกลับข้อมูลเมื่ออ่านไฟล์เสร็จ
      </p>
    </div>
  );
};

export default AsyncCallbacks;
