import React from "react";

const RecoilZustand = () => {
  return (
    <div className="max-w-3xl mx-auto p-6 shadow-lg rounded-lg border">
      <h1 className="text-2xl font-bold">Recoil & Zustand</h1>
      <p className="mt-4">
        <strong>Recoil</strong> และ <strong>Zustand</strong> เป็นตัวเลือกในการจัดการ State ที่เบาและใช้งานง่ายกว่า Redux
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่างการใช้ Zustand</h2>
      <pre className="p-4 rounded-md text-sm overflow-x-auto border bg-gray-200 dark:bg-gray-800">
{`import create from "zustand";

const useStore = create((set) => ({
  count: 0,
  increase: () => set((state) => ({ count: state.count + 1 }))
}));

const Counter = () => {
  const { count, increase } = useStore();
  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={increase}>เพิ่มค่า</button>
    </div>
  );
};`}
      </pre>
    </div>
  );
};

export default RecoilZustand;
