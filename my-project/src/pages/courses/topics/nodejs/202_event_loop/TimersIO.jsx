import React from "react";

const TimersIO = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold">⏳ Timers & I/O Operations</h1>
      <p className="mt-4">Node.js ใช้ **setTimeout, setInterval และ setImmediate** สำหรับการจัดการเวลา</p>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่าง Timers</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2">
        <code>{`setTimeout(() => console.log("Executed after 2s"), 2000);
setImmediate(() => console.log("Executed immediately"));`}</code>
      </pre>
    </div>
  );
};

export default TimersIO;
