import React from "react";

const Button = ({ text, onClick }) => {
  return (
    <button
      className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition"
      onClick={onClick}
    >
      {text}
    </button>
  );
};

const ReusableComponents = () => {
  return (
    <div className="max-w-3xl mx-auto p-6 shadow-lg rounded-lg border">
      <h1 className="text-2xl font-bold text-red-600">Reusable Components</h1>
      <p className="mt-4">
        <strong>Reusable Components</strong> คือ Components ที่สามารถนำกลับมาใช้ซ้ำได้ 
        โดยการใช้ Props เพื่อควบคุมพฤติกรรมของ Component
      </p>
      
      <h2 className="text-xl font-bold mt-6">✅ ตัวอย่างปุ่มที่ใช้ซ้ำได้</h2>
      <pre className="p-4 rounded-md text-sm overflow-x-auto border bg-gray-200 text-black dark:bg-gray-800 dark:text-white">
{`const Button = ({ text, onClick }) => {
  return <button onClick={onClick}>{text}</button>;
};`}
      </pre>
      
      <p className="mt-4">🔹 การใช้ปุ่มที่สร้างขึ้นมา</p>
      <pre className="p-4 rounded-md text-sm overflow-x-auto border bg-gray-200 text-black dark:bg-gray-800 dark:text-white">
{`const App = () => {
  return <Button text="Click Me" onClick={() => alert("Clicked!")} />;
};`}
      </pre>
      
      <p className="mt-4">
        การสร้าง Component ที่สามารถใช้ซ้ำได้ช่วยให้โค้ดสะอาดขึ้น และง่ายต่อการบำรุงรักษา
      </p>
      
      <div className="mt-6">
        <Button text="🔘 ตัวอย่างปุ่ม" onClick={() => alert("ปุ่มถูกกดแล้ว!")} />
      </div>
    </div>
  );
};

export default ReusableComponents;
