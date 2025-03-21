import React, { useEffect, useState } from "react";

const UseEffectHook = () => {
  const [count, setCount] = useState(0);

  useEffect(() => {
    document.title = `Count: ${count}`;
  }, [count]);

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold">useEffect & Side Effects</h1>

      <p className="mt-4 text-lg">
        <strong>useEffect</strong> เป็น Hook ที่ใช้จัดการกับ <em>side effects</em> ภายใน Functional Component
        เช่น การ fetch ข้อมูล, การเปลี่ยนแปลง DOM, การตั้งค่า timer หรือการ subscribe/unsubscribe
      </p>

      <h2 className="text-2xl font-semibold mt-6">📌 Syntax พื้นฐาน</h2>
      <pre className="p-4 mt-2 rounded-md overflow-x-auto border bg-gray-100 dark:bg-gray-800 text-black dark:text-white text-sm">
{`useEffect(() => {
  // ทำงานเมื่อ component mount หรือ update
  return () => {
    // cleanup เมื่อ component unmount (optional)
  };
}, [dependencies]);`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">🎯 ตัวอย่างการอัปเดต document.title</h2>
      <pre className="p-4 mt-2 rounded-md overflow-x-auto border bg-gray-100 dark:bg-gray-800 text-black dark:text-white text-sm">
{`const [count, setCount] = useState(0);

useEffect(() => {
  document.title = \`Count: \${count}\`;
}, [count]);`}
      </pre>

      <div className="mt-6">
        <button
          className="px-4 py-2 rounded bg-blue-600 hover:bg-blue-700 text-white transition"
          onClick={() => setCount(count + 1)}
        >
          เพิ่มค่า Count ({count})
        </button>
      </div>

      <p className="mt-6">
        ทุกครั้งที่กดปุ่ม <code>เพิ่มค่า</code> ค่า count จะเพิ่มขึ้นและ <code>document.title</code> ของเว็บจะเปลี่ยนตาม
      </p>
    </div>
  );
};

export default UseEffectHook;
