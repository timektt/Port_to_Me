import React from "react";

const WorkerThreads = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold">💼 Worker Threads</h1>
      <p className="mt-4">
        Node.js มี **Worker Threads** สำหรับรันโค้ดแบบ Parallel โดยไม่บล็อค Event Loop
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่าง Worker Threads</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2">
        <code>{`const { Worker } = require("worker_threads");

const worker = new Worker("./worker.js");

worker.on("message", (msg) => console.log("Worker:", msg));`}</code>
      </pre>
    </div>
  );
};

export default WorkerThreads;
