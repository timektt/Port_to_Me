import React from "react";

const ReduxBasics = () => {
  return (
    <div className="max-w-3xl mx-auto p-6">
      <h1 className="text-2xl font-bold">Redux Basics</h1>
      <p className="mt-4">
        <strong>Redux</strong> เป็น state management library ที่ใช้สำหรับจัดการ state ของแอปพลิเคชันในระดับ global
        โดยใช้หลักการ <strong>single source of truth</strong> คือมี store กลางที่เก็บข้อมูลทั้งหมดไว้ในที่เดียว
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 หลักการของ Redux</h2>
      <ul className="list-disc pl-6 mt-2 space-y-1">
        <li><strong>Store:</strong> เก็บ state ทั้งหมดไว้ในแอป</li>
        <li><strong>Action:</strong> ส่งข้อมูลเพื่อแจ้งว่าต้องการเปลี่ยนแปลงอะไร</li>
        <li><strong>Reducer:</strong> ระบุวิธีการเปลี่ยน state ตาม action ที่ได้รับ</li>
        <li><strong>Dispatch:</strong> ส่ง action ไปให้ reducer เพื่อให้เปลี่ยน state</li>
      </ul>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่าง Reducer และ Action</h2>
      <pre className="p-4 rounded-md text-sm overflow-x-auto border bg-gray-100 dark:bg-gray-800 dark:text-white">
{`const initialState = { count: 0 };

const counterReducer = (state = initialState, action) => {
  switch (action.type) {
    case "INCREMENT":
      return { count: state.count + 1 };
    case "DECREMENT":
      return { count: state.count - 1 };
    default:
      return state;
  }
};`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่างการใช้งาน Redux กับ React</h2>
      <pre className="p-4 rounded-md text-sm overflow-x-auto border bg-gray-100 dark:bg-gray-800 dark:text-white">
{`import { useDispatch, useSelector } from "react-redux";

const Counter = () => {
  const count = useSelector((state) => state.count);
  const dispatch = useDispatch();

  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => dispatch({ type: "INCREMENT" })}>เพิ่ม</button>
      <button onClick={() => dispatch({ type: "DECREMENT" })}>ลด</button>
    </div>
  );
};`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">📌 ข้อดีของ Redux</h2>
      <ul className="list-disc pl-6 mt-2 space-y-1">
        <li>แยก logic ออกจาก UI ได้ชัดเจน</li>
        <li>สามารถ debug และ track การเปลี่ยนแปลง state ได้ง่าย</li>
        <li>เหมาะกับแอปที่มี state ซับซ้อนและหลาย component ใช้ร่วมกัน</li>
      </ul>
    </div>
  );
};

export default ReduxBasics;
