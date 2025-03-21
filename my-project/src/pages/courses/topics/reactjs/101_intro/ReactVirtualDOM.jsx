import React from "react";

const ReactVirtualDOM = () => {
  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold">React Virtual DOM</h1>
      <p className="mt-4 text-lg">
        Virtual DOM เป็นกลไกที่ช่วยให้ React เรนเดอร์หน้าเว็บได้เร็วขึ้น โดยเปรียบเทียบการเปลี่ยนแปลงใน UI ก่อนอัปเดตจริง
      </p>

      <h2 className="text-2xl font-semibold mt-6">📌 วิธีทำงานของ Virtual DOM</h2>
      <ul className="list-disc list-inside mt-4 space-y-2">
        <li>React สร้าง Virtual DOM ที่เป็นโครงสร้างจำลองของ UI</li>
        <li>เมื่อข้อมูลเปลี่ยนแปลง React เปรียบเทียบ Virtual DOM กับ DOM จริง</li>
        <li>React อัปเดตเฉพาะส่วนที่เปลี่ยนแปลง</li>
      </ul>

      <h2 className="text-2xl font-semibold mt-6">🎯 ตัวอย่างการใช้ Virtual DOM</h2>
      <p className="mt-4">
        เมื่อใช้ React เราไม่ต้องแก้ไข DOM ตรง ๆ แต่ให้ React จัดการให้โดยใช้ <strong>state</strong> และ <strong>re-render</strong>
      </p>

      <pre className="p-4 rounded-md overflow-x-auto border bg-gray-200 text-black dark:bg-gray-800 dark:text-white">
        <code>{`import React, { useState } from "react";

const Counter = () => {
  const [count, setCount] = useState(0);

  return (
    <button onClick={() => setCount(count + 1)}>
      Count: {count}
    </button>
  );
};`}</code>
      </pre>
    </div>
  );
};

export default ReactVirtualDOM;
