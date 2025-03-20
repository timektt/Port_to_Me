import React from "react";

const NodeModules = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-screen-lg mx-auto">
      <h1 className="text-2xl sm:text-3xl font-bold">📦 โมดูลใน Node.js</h1>
      <p className="mt-4">
        Node.js ใช้ <strong>Modules</strong> ในการจัดการโค้ดให้สามารถแบ่งเป็นส่วน ๆ ได้ 
        โดยมีประเภทหลักดังนี้:
      </p>
      <ul className="list-disc ml-5 mt-2">
        <li><strong>Core Modules</strong>: โมดูลที่มาพร้อมกับ Node.js เช่น <code>fs</code>, <code>http</code>, <code>path</code></li>
        <li><strong>Custom Modules</strong>: โมดูลที่ผู้ใช้สร้างเอง</li>
        <li><strong>Third-party Modules</strong>: โมดูลจาก npm เช่น <code>express</code>, <code>mongoose</code></li>
      </ul>
      
      <h2 className="text-xl font-semibold mt-6">🔹 ตัวอย่างการใช้โมดูล fs (File System)</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        <code>{`const fs = require('fs');

fs.writeFileSync('test.txt', 'Hello from Node.js!');
console.log("ไฟล์ถูกสร้างเรียบร้อย");`}</code>
      </pre>
      
      <h2 className="text-xl font-semibold mt-6">🔹 การใช้โมดูล http (สร้างเว็บเซิร์ฟเวอร์)</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        <code>{`const http = require('http');

const server = http.createServer((req, res) => {
    res.writeHead(200, {'Content-Type': 'text/plain'});
    res.end('Hello from Node.js Server!');
});

server.listen(3000, () => {
    console.log('Server is running on http://localhost:3000');
});`}</code>
      </pre>
      
      <h2 className="text-xl font-semibold mt-6">🔹 การสร้าง Custom Module</h2>
      <p className="mt-2">สามารถสร้างโมดูลของตัวเองและนำไปใช้ในไฟล์อื่นได้</p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        <code>{`// myModule.js
exports.sayHello = function(name) {
    return ` + "`Hello, ${name}!`" + `;
};`}</code>
      </pre>
      
      <h2 className="text-xl font-semibold mt-6">🔹 การใช้โมดูลที่สร้างขึ้นเอง</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        <code>{`const myModule = require('./myModule');
console.log(myModule.sayHello('Node.js'));`}</code>
      </pre>
    </div>
  );
};

export default NodeModules;
