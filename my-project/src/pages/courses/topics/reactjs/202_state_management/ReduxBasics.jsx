import React from "react";

const ReduxBasics = () => {
  return (
    <div className="max-w-3xl mx-auto p-6 shadow-lg rounded-lg border">
      <h1 className="text-2xl font-bold">Redux Basics</h1>
      <p className="mt-4">
        <strong>Redux</strong> เป็นเครื่องมือจัดการ State ที่ใช้แนวคิด Centralized State เพื่อให้สามารถจัดการข้อมูลในแอปพลิเคชันได้ง่ายขึ้น
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่างโครงสร้าง Redux</h2>
      <pre className="p-4 rounded-md text-sm overflow-x-auto border bg-gray-200 dark:bg-gray-800">
{`const initialState = { count: 0 };

const counterReducer = (state = initialState, action) => {
  switch (action.type) {
    case "INCREMENT":
      return { count: state.count + 1 };
    default:
      return state;
  }
};`}
      </pre>
    </div>
  );
};

export default ReduxBasics;
