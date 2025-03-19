import React from "react";

const Card = ({ title, children }) => {
  return (
    <div className="border p-4 rounded-lg shadow-lg bg-gray-100 dark:bg-gray-800">
      <h2 className="text-lg font-bold text-blue-600 dark:text-blue-400">{title}</h2>
      <div className="mt-2 text-gray-700 dark:text-gray-300">{children}</div>
    </div>
  );
};

const CompositionVsInheritance = () => {
  return (
    <div className="max-w-3xl mx-auto p-6">
      <h1 className="text-2xl font-bold text-indigo-600 dark:text-indigo-400">
        Composition vs Inheritance
      </h1>
      <p className="mt-4">
        React สนับสนุนแนวคิด <strong>Composition</strong> มากกว่า <strong>Inheritance</strong> 
        เพื่อให้การจัดการ Component ง่ายขึ้นและมีประสิทธิภาพมากขึ้น
      </p>

      <h2 className="text-xl font-bold mt-6">📌 การใช้ Composition</h2>
      <p className="mt-2">Composition ช่วยให้เราสามารถรวม Component ได้ง่ายขึ้น</p>
      <pre className="p-4 mt-2 rounded-md overflow-x-auto border bg-gray-200 text-black dark:bg-gray-800 dark:text-white">
{`const Card = ({ title, children }) => {
  return (
    <div className="border p-4 rounded-lg shadow-lg">
      <h2>{title}</h2>
      <div>{children}</div>
    </div>
  );
};

const App = () => {
  return (
    <Card title="React Composition">
      <p>นี่คือตัวอย่างของ Composition</p>
    </Card>
  );
};`}
      </pre>

      <h2 className="text-xl font-bold mt-6">📌 ข้อดีของ Composition</h2>
      <ul className="list-disc pl-6 mt-2">
        <li>ช่วยให้การจัดการ Component เป็นไปอย่างมีประสิทธิภาพ</li>
        <li>สามารถรวม Component ได้ง่ายและยืดหยุ่นกว่า</li>
        <li>ลดการใช้ Inheritance ที่ซับซ้อนเกินไป</li>
      </ul>

      <Card title="ตัวอย่าง Composition">
        <p>React แนะนำให้ใช้ Composition มากกว่า Inheritance</p>
      </Card>
    </div>
  );
};

export default CompositionVsInheritance;
