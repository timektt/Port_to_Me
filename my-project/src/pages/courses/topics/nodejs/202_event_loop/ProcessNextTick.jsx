import React from "react";

const ProcessNextTick = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold">🚀 process.nextTick()</h1>
      <p className="mt-4">
        คำสั่ง <code>process.nextTick()</code> ใช้สำหรับรันโค้ดให้ทำงานก่อน Task อื่น ๆ ใน Event Loop
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 การทำงานของ process.nextTick()</h2>
      <p className="mt-2">
        <code>process.nextTick()</code> ช่วยให้โค้ดที่อยู่ภายในทำงานทันทีหลังจากฟังก์ชันปัจจุบันใน Call Stack ทำงานเสร็จ
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`console.log("Start");

process.nextTick(() => console.log("Next Tick"));

console.log("End");`}</code>
      </pre>
      <p className="mt-4">👉 ผลลัพธ์: <code>"Start" → "End" → "Next Tick"</code></p>

      <h2 className="text-xl font-semibold mt-6">🔄 เปรียบเทียบกับ setImmediate()</h2>
      <p className="mt-2">
        <code>process.nextTick()</code> รันก่อน <code>setImmediate()</code> เสมอ เนื่องจากมันถูกจัดให้อยู่ใน Microtask Queue
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
        <code>{`console.log("Start");

setImmediate(() => console.log("setImmediate"));
process.nextTick(() => console.log("Next Tick"));

console.log("End");`}</code>
      </pre>
      <p className="mt-4">👉 ผลลัพธ์: <code>"Start" → "End" → "Next Tick" → "setImmediate"</code></p>

      <h2 className="text-xl font-semibold mt-6">✅ สรุป</h2>
      <ul className="list-disc ml-5 mt-2">
        <li><strong>process.nextTick()</strong>: รันทันทีหลังจากฟังก์ชันปัจจุบันทำงานเสร็จ</li>
        <li><strong>setImmediate()</strong>: รันหลังจาก Event Loop รอบปัจจุบันเสร็จสมบูรณ์</li>
      </ul>
    </div>
  );
};

export default ProcessNextTick;