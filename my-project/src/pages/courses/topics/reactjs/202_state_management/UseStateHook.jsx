import React, { useState } from "react";

const UseStateHook = () => {
  const [count, setCount] = useState(0);

  return (
    <div className="max-w-3xl mx-auto p-6 shadow-lg rounded-lg border">
      <h1 className="text-2xl font-bold">Using useState Hook</h1>
      <p className="mt-4">
        <strong>useState</strong> เป็น Hook ที่ใช้จัดการ State ใน Functional Components ทำให้เราสามารถอัปเดตค่า State ได้โดยง่าย
      </p>
      
      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่างการใช้งาน</h2>
      <pre className="p-4 rounded-md text-sm overflow-x-auto border bg-gray-200 dark:bg-gray-800">
{`const [count, setCount] = useState(0);

return (
  <div>
    <p>Count: {count}</p>
    <button onClick={() => setCount(count + 1)}>เพิ่มค่า</button>
  </div>
);`}
      </pre>

      <div className="mt-6">
        <p>ค่า Count ปัจจุบัน: {count}</p>
        <button 
          className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition mt-2"
          onClick={() => setCount(count + 1)}
        >
          เพิ่มค่า
        </button>
      </div>
    </div>
  );
};

export default UseStateHook;
