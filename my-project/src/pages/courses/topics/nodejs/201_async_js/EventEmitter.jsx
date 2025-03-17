import React from "react";

const EventEmitter = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold">🔔 EventEmitter in Node.js</h1>

      <p className="mt-4 text-lg">
        ใน Node.js โมดูล `events` ถูกใช้ในการจัดการเหตุการณ์ (Event Handling) โดยมี **EventEmitter** เป็นหัวใจหลัก
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่างการใช้ EventEmitter</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2">
        <code>{`const EventEmitter = require('events');
const myEmitter = new EventEmitter();

myEmitter.on('greet', (name) => {
  console.log(\`Hello, \${name}!\`);
});

myEmitter.emit('greet', 'Supermhee');`}</code>
      </pre>

      <p className="mt-4">
        ในโค้ดนี้ เราสร้างอ็อบเจ็กต์ `myEmitter` และกำหนดให้เมื่อมีการเรียก Event `'greet'` จะแสดงข้อความออกทาง console
      </p>
    </div>
  );
};

export default EventEmitter;
