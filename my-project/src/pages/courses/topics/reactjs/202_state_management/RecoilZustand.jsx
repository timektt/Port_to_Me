import React from "react";

const RecoilZustand = () => {
  return (
    <div className="max-w-3xl mx-auto p-6">
      <h1 className="text-2xl font-bold">Recoil & Zustand</h1>
      <p className="mt-4">
        การจัดการ State เป็นหัวใจหลักของแอปพลิเคชัน React โดย <strong>Zustand</strong> และ <strong>Recoil</strong> 
        เป็นไลบรารีที่ได้รับความนิยม เนื่องจากใช้งานง่าย เบา และไม่ต้องเขียนโค้ดซ้ำซ้อนแบบ Redux
      </p>

      <h2 className="text-xl font-semibold mt-6">✅ Zustand คืออะไร?</h2>
      <p className="mt-2">
        Zustand เป็นไลบรารีที่สร้าง Global Store ได้อย่างง่ายดายโดยไม่ต้องใช้ Context API หรือ Reducer 
        และยังรองรับ TypeScript, Middleware, Persist store ได้ด้วย
      </p>

      <h3 className="text-lg font-semibold mt-4">📌 ตัวอย่าง Zustand</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md text-sm overflow-x-auto">
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

      <h2 className="text-xl font-semibold mt-8">✅ Recoil คืออะไร?</h2>
      <p className="mt-2">
        Recoil เป็นไลบรารีจากทีม Facebook ที่ออกแบบมาเพื่อให้จัดการ state ได้เหมือน React-native แต่รองรับการใช้งานในระดับ Component
        โดยใช้แนวคิด <strong>Atoms</strong> และ <strong>Selectors</strong>
      </p>

      <h3 className="text-lg font-semibold mt-4">📌 ตัวอย่าง Recoil</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md text-sm overflow-x-auto">
{`import { atom, useRecoilState, RecoilRoot } from "recoil";

const countState = atom({
  key: "countState",
  default: 0,
});

const Counter = () => {
  const [count, setCount] = useRecoilState(countState);
  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>เพิ่มค่า</button>
    </div>
  );
};

const App = () => (
  <RecoilRoot>
    <Counter />
  </RecoilRoot>
);`}
      </pre>

      <h2 className="text-xl font-semibold mt-8">⚖️ เปรียบเทียบ Zustand vs Recoil</h2>
      <ul className="list-disc pl-6 space-y-2 mt-2">
        <li><strong>Zustand</strong>: ใช้งานง่าย ไม่ต้องใช้ Provider ครอบ</li>
        <li><strong>Recoil</strong>: ทำงานคล้าย React, ควบคุมการไหลของ State ได้ละเอียด</li>
        <li>Zustand เบากว่า และเหมาะกับโปรเจกต์ที่ต้องการความเร็วและโค้ดน้อย</li>
        <li>Recoil เหมาะกับแอปขนาดใหญ่ที่ต้องการแบ่ง State ชัดเจน</li>
      </ul>
    </div>
  );
};

export default RecoilZustand;
