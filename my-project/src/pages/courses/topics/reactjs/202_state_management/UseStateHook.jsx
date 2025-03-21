import React, { useState } from "react";

const UseStateHook = () => {
  const [count, setCount] = useState(0);

  return (
    <div className="max-w-3xl mx-auto p-6">
      <h1 className="text-2xl font-bold">ใช้ useState Hook ใน React</h1>
      <p className="mt-4">
        <strong>useState</strong> คือ React Hook ที่ใช้สำหรับกำหนดและจัดการ state ในฟังก์ชันคอมโพเนนต์ 
        โดยจะคืนค่าออกมาเป็น array ที่มี 2 ค่า คือ state ปัจจุบัน และฟังก์ชันสำหรับอัปเดต state นั้น
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 โครงสร้างพื้นฐาน</h2>
      <pre className="p-4 rounded-md text-sm overflow-x-auto border bg-gray-100 dark:bg-gray-800 dark:text-white">
{`const [state, setState] = useState(initialValue);`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่างการใช้งาน</h2>
      <pre className="p-4 rounded-md text-sm overflow-x-auto border bg-gray-100 dark:bg-gray-800 dark:text-white">
{`const [count, setCount] = useState(0);

return (
  <div>
    <p>Count: {count}</p>
    <button onClick={() => setCount(count + 1)}>เพิ่มค่า</button>
  </div>
);`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">📌 ผลลัพธ์ที่ได้จากการคลิก</h2>
      <p className="mt-2">
        เมื่อคลิกปุ่ม <code>เพิ่มค่า</code> ค่าของ <code>count</code> จะเพิ่มขึ้นทีละ 1 โดยที่คอมโพเนนต์จะ re-render เพื่อแสดงผลใหม่ทันที
      </p>

      <div className="mt-6 border rounded-md p-4">
        <p className="text-lg font-medium">ค่าปัจจุบัน: <strong>{count}</strong></p>
        <button 
          className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg mt-2 transition"
          onClick={() => setCount(count + 1)}
        >
          เพิ่มค่า
        </button>
      </div>

      <h2 className="text-xl font-semibold mt-6">📌 สรุป</h2>
      <ul className="list-disc pl-6 mt-2 space-y-1">
        <li><code>useState</code> ใช้เพื่อเพิ่มความสามารถในการจัดการข้อมูลภายในคอมโพเนนต์</li>
        <li>เมื่อมีการเปลี่ยน state, React จะทำการ re-render คอมโพเนนต์</li>
        <li>ใช้ได้เฉพาะในฟังก์ชันคอมโพเนนต์ (ไม่ใช้กับ class)</li>
      </ul>
    </div>
  );
};

export default UseStateHook;
