import React from "react";

const StreamsBuffer = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto overflow-x-hidden">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold break-words">
        📡 Streams & Buffers
      </h1>

      <p className="mt-4 text-lg break-words">
        <strong>Streams</strong> ใน Node.js ใช้สำหรับการอ่าน/เขียนข้อมูลขนาดใหญ่ เช่น ไฟล์ หรือ Network Request โดยไม่ต้องโหลดทั้งหมดเข้าหน่วยความจำ
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่างการใช้ Readable Stream</h2>
      <div className="overflow-x-auto">
        <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 text-sm sm:text-base">
          <code>{`const fs = require('fs');
const readStream = fs.createReadStream('largefile.txt', 'utf8');

readStream.on('data', (chunk) => {
  console.log('Received chunk:', chunk);
});`}</code>
        </pre>
      </div>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่างการใช้ Writable Stream</h2>
      <div className="overflow-x-auto">
        <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 text-sm sm:text-base">
          <code>{`const writeStream = fs.createWriteStream('output.txt');

writeStream.write('Hello, World!');
writeStream.end();`}</code>
        </pre>
      </div>

      <p className="mt-4 break-words">
        Streams ช่วยให้ Node.js จัดการข้อมูลขนาดใหญ่ได้อย่างมีประสิทธิภาพโดยไม่ต้องโหลดทั้งหมดในครั้งเดียว
      </p>

      <h2 className="text-xl font-semibold mt-6">🎯 ข้อดีของ Streams & Buffers</h2>
      <ul className="list-disc ml-6 mt-2 space-y-1 text-base">
        <li>ประหยัดหน่วยความจำ — อ่านทีละส่วน ไม่โหลดทั้งหมด</li>
        <li>ประสิทธิภาพสูงสำหรับข้อมูลขนาดใหญ่</li>
        <li>เหมาะกับการส่งไฟล์, สตรีมเสียง/วิดีโอ, หรือประมวลผลต่อเนื่อง</li>
      </ul>

      <h2 className="text-xl font-semibold mt-6">⚠️ สิ่งที่ควรระวัง</h2>
      <ul className="list-disc ml-6 mt-2 space-y-1 text-base">
        <li>ต้องจัดการ event ให้ดี เช่น 'error', 'end'</li>
        <li>อาจต้องใช้ pipe เพื่อความสะดวกในการเชื่อมต่อ stream</li>
        <li>Buffer ที่มากเกินไปอาจทำให้หน่วยความจำเต็มได้</li>
      </ul>
    </div>
  );
};

export default StreamsBuffer;
