import React from "react";

const TimersIO = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold">⏳ Timers & I/O Operations</h1>
      <p className="mt-4">Node.js ใช้ <strong>setTimeout, setInterval และ setImmediate</strong> สำหรับการจัดการเวลา</p>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่าง Timers</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`setTimeout(() => console.log("Executed after 2s"), 2000);
setImmediate(() => console.log("Executed immediately"));
setInterval(() => console.log("Repeated every 3s"), 3000);`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">🔄 การทำงานของ Timers ใน Event Loop</h2>
      <p className="mt-2">
        <strong>Timers</strong> ใน Node.js ทำงานผ่าน Event Loop ซึ่งมีลำดับดังนี้:
      </p>
      <ul className="list-disc ml-5 mt-2">
        <li><code>setTimeout()</code>: ทำงานเมื่อเวลาที่กำหนดหมดลง</li>
        <li><code>setInterval()</code>: ทำงานซ้ำทุกช่วงเวลาที่กำหนด</li>
        <li><code>setImmediate()</code>: ทำงานหลังจาก I/O Callback ถูกดำเนินการ</li>
      </ul>

      <h2 className="text-xl font-semibold mt-6">📌 เปรียบเทียบ setImmediate() และ setTimeout()</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`setTimeout(() => console.log("Timeout after 0ms"), 0);
setImmediate(() => console.log("Immediate execution"));`}</code>
      </pre>
      <p className="mt-4">👉 ผลลัพธ์ขึ้นอยู่กับสภาวะของ Event Loop แต่โดยทั่วไป <code>setImmediate()</code> จะทำงานก่อน <code>setTimeout()</code> ที่มีค่าเป็น 0ms</p>

      <h2 className="text-xl font-semibold mt-6">✅ สรุป</h2>
      <ul className="list-disc ml-5 mt-2">
        <li><strong>setTimeout()</strong>: รอเวลาที่กำหนดแล้วจึงรัน</li>
        <li><strong>setInterval()</strong>: ทำงานซ้ำทุกช่วงเวลาที่กำหนด</li>
        <li><strong>setImmediate()</strong>: ทำงานทันทีหลังจาก Event Loop ดำเนินการเสร็จ</li>
      </ul>
    </div>
  );
};

export default TimersIO;