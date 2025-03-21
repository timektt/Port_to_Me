import React, { useRef } from "react";

const UseRefHook = () => {
  const inputRef = useRef(null);

  const focusInput = () => {
    inputRef.current.focus();
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold">useRef & Manipulating DOM</h1>

      <p className="mt-4 text-lg">
        <code>useRef</code> เป็น Hook ที่ใช้เก็บค่าหรืออ้างอิง DOM Element ได้โดยไม่ทำให้ Component re-render เมื่อค่าเปลี่ยน
      </p>

      <h2 className="text-2xl font-semibold mt-6">📌 การอ้างอิง DOM Element</h2>
      <pre className="p-4 mt-2 rounded-md overflow-x-auto border bg-gray-100 dark:bg-gray-800 text-black dark:text-white text-sm">
{`const inputRef = useRef(null);

<input ref={inputRef} />`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">📌 การโฟกัส Element</h2>
      <pre className="p-4 mt-2 rounded-md overflow-x-auto border bg-gray-100 dark:bg-gray-800 text-black dark:text-white text-sm">
{`const focusInput = () => {
  inputRef.current.focus();
};`}
      </pre>

      <input
        ref={inputRef}
        className="p-2 border rounded mt-4 block text-black"
        placeholder="พิมพ์อะไรสักอย่าง..."
      />
      <button
        className="mt-4 p-2 border rounded bg-green-500 text-gray-800 hover:bg-green-600 transition"
        onClick={focusInput}
      >
        โฟกัส Input
      </button>

      <p className="mt-6">
        โค้ดนี้จะทำให้เราสามารถอ้างอิงไปยัง input แล้วเรียก focus ได้โดยตรงผ่าน <code>useRef</code> โดยไม่ต้องพึ่งการจัดการผ่าน state
      </p>
    </div>
  );
};

export default UseRefHook;
