import React from "react";

const NodeIntro = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-screen-lg mx-auto">
      <h1 className="text-2xl sm:text-3xl font-bold">🚀 แนะนำ Node.js</h1>
      <p className="mt-4">
        Node.js เป็น JavaScript runtime ที่ช่วยให้สามารถรัน JavaScript นอก Web Browser ได้ 
        โดยใช้ V8 Engine ที่พัฒนาโดย Google ทำให้สามารถใช้ JavaScript สำหรับพัฒนา Backend ได้อย่างเต็มประสิทธิภาพ
      </p>
      
      <h2 className="text-xl font-semibold mt-6">🔹 คุณสมบัติสำคัญของ Node.js</h2>
      <ul className="list-disc ml-5 mt-2">
        <li><strong>Non-blocking I/O</strong> - รองรับการทำงานแบบ asynchronous ทำให้ประสิทธิภาพสูง</li>
        <li><strong>Event-driven Architecture</strong> - ใช้แนวคิด event loop ในการประมวลผล</li>
        <li><strong>Single-threaded</strong> - ทำงานบนเธรดเดียวแต่สามารถจัดการงานได้หลายอย่างพร้อมกัน</li>
        <li><strong>ใช้ V8 Engine</strong> - ทำให้ JavaScript ทำงานได้เร็วและมีประสิทธิภาพ</li>
      </ul>
      
      <h2 className="text-xl font-semibold mt-6">🔹 ติดตั้ง Node.js</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        {`# สำหรับ macOS และ Linux
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# สำหรับ Windows
ดาวน์โหลดตัวติดตั้งจาก https://nodejs.org/`}
      </pre>
      
      <h2 className="text-xl font-semibold mt-6">🔹 ตรวจสอบเวอร์ชันของ Node.js</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        {`node -v`}
      </pre>
      
      <h2 className="text-xl font-semibold mt-6">🔹 ตัวอย่างการรัน Node.js</h2>
      <p className="mt-2">สามารถรัน JavaScript บน Node.js ได้ง่าย ๆ ดังนี้</p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        {`console.log("Hello from Node.js");`}
      </pre>
      
      <h2 className="text-xl font-semibold mt-6">🔹 การสร้างเซิร์ฟเวอร์ง่าย ๆ ด้วย Node.js</h2>
      <p className="mt-2">Node.js สามารถสร้าง Web Server ได้โดยใช้ `http` module</p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        {`const http = require('http');

const server = http.createServer((req, res) => {
    res.writeHead(200, {'Content-Type': 'text/plain'});
    res.end('Hello, Node.js!');
});

server.listen(3000, () => {
    console.log('Server is running on http://localhost:3000');
});`}
      </pre>
    </div>
  );
};

export default NodeIntro;