import React from "react";

const NodeJsRunCode = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-screen-lg mx-auto">
      <h1 className="text-2xl sm:text-3xl font-bold">🚀 รัน JavaScript ใน Node.js</h1>
      <p className="mt-4">
        สามารถรันโค้ด JavaScript บน Node.js ได้โดยใช้ Terminal หรือ Command Line ทำให้เราสามารถพัฒนา Backend ได้โดยไม่ต้องใช้ Web Browser
      </p>
      
      <h2 className="text-xl font-semibold mt-6">🔹 ติดตั้ง Node.js (ถ้ายังไม่ได้ติดตั้ง)</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        {`# สำหรับ macOS และ Linux
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# สำหรับ Windows
ดาวน์โหลดตัวติดตั้งจาก https://nodejs.org/ และติดตั้งตามขั้นตอน`}
      </pre>
      
      <h2 className="text-xl font-semibold mt-6">🔹 ตรวจสอบเวอร์ชันของ Node.js</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        {`node -v`}
      </pre>
      
      <h2 className="text-xl font-semibold mt-6">🔹 สร้างไฟล์ JavaScript</h2>
      <p className="mt-2">สร้างไฟล์ JavaScript ที่ต้องการรัน เช่น `app.js` และเพิ่มโค้ดดังนี้</p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        {`// app.js
console.log("Hello from Node.js");`}
      </pre>
      
      <h2 className="text-xl font-semibold mt-6">🔹 รันโค้ดใน Terminal</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        {`node app.js`}
      </pre>
      
      <h2 className="text-xl font-semibold mt-6">🔹 การใช้ Node.js เพื่อรัน JavaScript แบบ Interactive</h2>
      <p className="mt-2">Node.js มี REPL (Read-Eval-Print Loop) ที่ช่วยให้สามารถรัน JavaScript ได้แบบโต้ตอบ</p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        {`# เปิดโหมด Interactive
node

# ทดสอบคำสั่ง
> console.log("Hello, Node.js!");
Hello, Node.js!`}
      </pre>
      
      <h2 className="text-xl font-semibold mt-6">🔹 การสร้างเซิร์ฟเวอร์ HTTP ง่าย ๆ ด้วย Node.js</h2>
      <p className="mt-2">สามารถใช้โมดูล `http` เพื่อสร้างเซิร์ฟเวอร์แบบง่าย ๆ</p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        {`// server.js
const http = require('http');

const server = http.createServer((req, res) => {
    res.writeHead(200, {'Content-Type': 'text/plain'});
    res.end('Hello, Node.js Server!');
});

server.listen(3000, () => {
    console.log('Server is running on http://localhost:3000');
});`}
      </pre>
    </div>
  );
};

export default NodeJsRunCode;
