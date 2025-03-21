import React, { useReducer } from "react";

// ✅ Reducer function สำหรับอัปเดต state ตาม action
const reducer = (state, action) => {
  switch (action.type) {
    case "increment":
      return { count: state.count + 1 };
    case "decrement":
      return { count: state.count - 1 };
    default:
      return state;
  }
};

const UseReducerHook = () => {
  const [state, dispatch] = useReducer(reducer, { count: 0 });

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold">useReducer & State Management</h1>

      <p className="mt-4 text-lg">
        <strong>useReducer</strong> เป็น Hook ที่เหมาะสำหรับการจัดการ state ที่มีความซับซ้อน หรือมีหลาย action
        โดยใช้แนวคิดคล้าย Redux เช่น แยก logic การเปลี่ยน state ออกมาเป็นฟังก์ชัน reducer
      </p>

      <h2 className="text-2xl font-semibold mt-6">📌 โครงสร้างพื้นฐานของ useReducer</h2>
      <pre className="p-4 mt-2 rounded-md overflow-x-auto border bg-gray-100 dark:bg-gray-800 text-black dark:text-white text-sm">
{`const [state, dispatch] = useReducer(reducer, initialState);`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">📌 ตัวอย่าง reducer function</h2>
      <pre className="p-4 mt-2 rounded-md overflow-x-auto border bg-gray-100 dark:bg-gray-800 text-black dark:text-white text-sm">
{`const reducer = (state, action) => {
  switch (action.type) {
    case "increment":
      return { count: state.count + 1 };
    case "decrement":
      return { count: state.count - 1 };
    default:
      return state;
  }
};`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">🎯 ตัวอย่างการใช้งาน</h2>
      <pre className="p-4 mt-2 rounded-md overflow-x-auto border bg-gray-100 dark:bg-gray-800 text-black dark:text-white text-sm">
{`const [state, dispatch] = useReducer(reducer, { count: 0 });

return (
  <div>
    <button onClick={() => dispatch({ type: "decrement" })}>-</button>
    <span>{state.count}</span>
    <button onClick={() => dispatch({ type: "increment" })}>+</button>
  </div>
);`}
      </pre>

      <div className="mt-6 flex gap-4 items-center">
        <button
          className="px-4 py-2 rounded bg-red-600 hover:bg-red-700 text-white transition"
          onClick={() => dispatch({ type: "decrement" })}
        >
          - ลด
        </button>
        <span className="text-xl font-semibold">{state.count}</span>
        <button
          className="px-4 py-2 rounded bg-green-600 hover:bg-green-700 text-white transition"
          onClick={() => dispatch({ type: "increment" })}
        >
          + เพิ่ม
        </button>
      </div>

      <p className="mt-6">
        โค้ดนี้ใช้ <code>dispatch</code> เพื่อส่ง action ไปให้ <code>reducer</code> ซึ่งจะอัปเดต state ให้ใหม่โดยไม่ต้องจัดการหลาย state ด้วยตัวเอง
      </p>
    </div>
  );
};

export default UseReducerHook;
