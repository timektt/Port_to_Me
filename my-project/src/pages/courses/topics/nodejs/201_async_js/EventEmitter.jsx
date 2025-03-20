import React from "react";

const EventEmitter = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold">🔔 EventEmitter in Node.js</h1>

      <p className="mt-4 text-lg">
        ใน Node.js โมดูล <code>events</code> ถูกใช้ในการจัดการเหตุการณ์ (Event Handling) โดยมี <strong>EventEmitter</strong> เป็นหัวใจหลัก
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่างการใช้ EventEmitter</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`const EventEmitter = require('events');
const myEmitter = new EventEmitter();

myEmitter.on('greet', (name) => {
  console.log(\`Hello, \${name}!\`);
});

myEmitter.emit('greet', 'Superbear');`}</code>
      </pre>

      <p className="mt-4">
        ในโค้ดนี้ เราสร้างอ็อบเจ็กต์ <code>myEmitter</code> และกำหนดให้เมื่อมีการเรียก Event <code>'greet'</code> จะแสดงข้อความออกทาง console
      </p>

      <h2 className="text-xl font-semibold mt-6">🔄 การใช้ EventEmitter กับการ Delay</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`const EventEmitter = require('events');
const myEmitter = new EventEmitter();

myEmitter.on('processComplete', () => {
  console.log('Processing complete!');
});

console.log('Starting process...');
setTimeout(() => {
  myEmitter.emit('processComplete');
}, 2000);`}</code>
      </pre>
      <p className="mt-4">
        ตัวอย่างนี้แสดงให้เห็นว่า EventEmitter สามารถใช้ร่วมกับ <code>setTimeout</code> เพื่อทำให้เหตุการณ์ถูกเรียกหลังจากเวลาที่กำหนด
      </p>

      <h2 className="text-xl font-semibold mt-6">🚀 การส่งข้อมูลเพิ่มเติมผ่าน Event</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`myEmitter.on('orderPlaced', (order) => {
  console.log(\`Order received: \${order.item} - Quantity: \${order.quantity}\`);
});

myEmitter.emit('orderPlaced', { item: 'Laptop', quantity: 2 });`}</code>
      </pre>
      <p className="mt-4">
        ตัวอย่างนี้แสดงการส่ง Object ผ่าน EventEmitter เพื่อให้สามารถจัดการข้อมูลได้ยืดหยุ่นขึ้น
      </p>
    </div>
  );
};

export default EventEmitter;
