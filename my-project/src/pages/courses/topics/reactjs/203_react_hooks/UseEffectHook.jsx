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
        useEffect เป็น Hook ที่ใช้สำหรับจัดการ Side Effects เช่น Fetching Data, Manipulating DOM เป็นต้น
      </p>
      <button className="mt-4 p-2 border rounded bg-blue-500 text-white" onClick={() => setCount(count + 1)}>
        เพิ่มค่า Count ({count})
      </button>
    </div>
  );
};

export default UseEffectHook;
