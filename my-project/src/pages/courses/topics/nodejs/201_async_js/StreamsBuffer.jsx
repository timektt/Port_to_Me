import React from "react";

const StreamsBuffer = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold">📡 Streams & Buffers</h1>

      <p className="mt-4 text-lg">
        **Streams** ใน Node.js ใช้สำหรับการอ่าน/เขียนข้อมูลขนาดใหญ่ เช่น ไฟล์ หรือ Network Request โดยไม่ต้องโหลดทั้งหมดเข้าหน่วยความจำ
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่างการใช้ Readable Stream</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2">
        <code>{`const fs = require('fs');
const readStream = fs.createReadStream('largefile.txt', 'utf8');

readStream.on('data', (chunk) => {
  console.log('Received chunk:', chunk);
});`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่างการใช้ Writable Stream</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2">
        <code>{`const writeStream = fs.createWriteStream('output.txt');

writeStream.write('Hello, World!');
writeStream.end();`}</code>
      </pre>

      <p className="mt-4">
        Streams ช่วยให้ Node.js จัดการข้อมูลขนาดใหญ่ได้อย่างมีประสิทธิภาพโดยไม่ต้องโหลดทั้งหมดในครั้งเดียว
      </p>
    </div>
  );
};

export default StreamsBuffer;
