import React from "react";

const ProcessNextTick = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold">🚀 process.nextTick()</h1>
      <p className="mt-4">
        คำสั่ง **process.nextTick()** ใช้สำหรับรันโค้ดให้ทำงานก่อน Task อื่น ๆ ใน Event Loop
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่าง process.nextTick()</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2">
        <code>{`console.log("Start");

process.nextTick(() => console.log("Next Tick"));

console.log("End");`}</code>
      </pre>

      <p className="mt-4">👉 ผลลัพธ์: `"Start" → "End" → "Next Tick"`</p>
    </div>
  );
};

export default ProcessNextTick;
