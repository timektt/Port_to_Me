import React from "react";

const WorkerThreads = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold">💼 Worker Threads</h1>
      <p className="mt-4">
        Node.js มี <strong>Worker Threads</strong> สำหรับรันโค้ดแบบ Parallel โดยไม่บล็อค Event Loop ทำให้สามารถใช้ประโยชน์จาก Multi-Core CPU ได้ดีขึ้น
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 วิธีใช้งาน Worker Threads</h2>
      <p className="mt-2">เราสามารถสร้าง Worker Thread เพื่อรันโค้ดแยกจาก Main Thread ได้ดังนี้:</p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`// main.js
const { Worker } = require("worker_threads");

const worker = new Worker("./worker.js");

worker.on("message", (msg) => console.log("Worker:", msg));
worker.on("error", (err) => console.error("Worker Error:", err));
worker.on("exit", (code) => console.log("Worker exited with code", code));`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่างโค้ดใน worker.js</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`// worker.js
const { parentPort } = require("worker_threads");

let count = 0;
setInterval(() => {
  count++;
  parentPort.postMessage(\`Worker is running: Count = \${count}\`);
}, 1000);`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">📌 ทำไมต้องใช้ Worker Threads?</h2>
      <ul className="list-disc ml-5 mt-2">
        <li>เหมาะสำหรับงานประมวลผลหนัก เช่น การคำนวณและ Machine Learning</li>
        <li>ช่วยให้ Node.js ใช้ CPU Multi-Core ได้เต็มประสิทธิภาพ</li>
        <li>ลดการบล็อก Event Loop และเพิ่มประสิทธิภาพของแอปพลิเคชัน</li>
      </ul>

      <h2 className="text-xl font-semibold mt-6">✅ สรุป</h2>
      <p className="mt-2">
        <strong>Worker Threads</strong> เป็นฟีเจอร์ที่ช่วยให้ Node.js ทำงานแบบ Multi-Threading ได้อย่างมีประสิทธิภาพ โดยเหมาะกับงานที่ต้องใช้การคำนวณหนักหรือการทำงานแบบ Parallel
      </p>
    </div>
  );
};

export default WorkerThreads;