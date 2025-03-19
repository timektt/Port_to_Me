import React, { useReducer } from "react";

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
      <p className="mt-4 text-lg">useReducer ใช้สำหรับจัดการ state ที่ซับซ้อนกว่าการใช้ useState</p>
      <div className="mt-4 flex gap-4">
        <button className="p-2 border rounded bg-red-500 text-white" onClick={() => dispatch({ type: "decrement" })}>
          - ลด
        </button>
        <span className="p-2">{state.count}</span>
        <button className="p-2 border rounded bg-green-500 text-white" onClick={() => dispatch({ type: "increment" })}>
          + เพิ่ม
        </button>
      </div>
    </div>
  );
};

export default UseReducerHook;