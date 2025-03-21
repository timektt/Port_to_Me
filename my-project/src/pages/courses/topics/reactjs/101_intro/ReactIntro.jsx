import React from "react";

const ReactIntro = () => {
  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold">React.js คืออะไร?</h1>
      <p className="mt-4 text-lg">
        React.js เป็น JavaScript Library ที่ใช้ในการพัฒนา UI ของเว็บแอปพลิเคชัน โดย Facebook เป็นผู้พัฒนา
        โดดเด่นด้วยความสามารถในการจัดการ Component และเรนเดอร์ UI อย่างมีประสิทธิภาพผ่าน Virtual DOM
      </p>

      <h2 className="text-2xl font-semibold mt-6">📌 คุณสมบัติหลักของ React.js</h2>
      <ul className="list-disc list-inside mt-4 space-y-2">
        <li>✅ <strong>Component-based:</strong> แยกโค้ดออกเป็นชิ้น ๆ สามารถนำกลับมาใช้ซ้ำได้</li>
        <li>✅ <strong>Virtual DOM:</strong> เพิ่มประสิทธิภาพการอัปเดตหน้าจอ</li>
        <li>✅ <strong>Declarative:</strong> บอก React ว่า UI ควรเป็นอย่างไร</li>
        <li>✅ <strong>Unidirectional Data Flow:</strong> ข้อมูลไหลจากบนลงล่างผ่าน Props</li>
      </ul>

      <h2 className="text-2xl font-semibold mt-6">⚙️ JSX คืออะไร?</h2>
      <p className="mt-2">
        JSX เป็นไวยากรณ์ที่คล้าย HTML แต่ใช้ใน JavaScript เพื่อสร้าง UI โดยตรง:
      </p>
      <pre className="p-4 mt-2 rounded-lg overflow-x-auto text-sm font-mono border">
{`const element = <h1>Hello JSX!</h1>;`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">🎯 ตัวอย่างการสร้าง Functional Component</h2>
      <pre className="p-4 mt-2 rounded-lg overflow-x-auto text-sm font-mono border">
{`const MyComponent = () => {
  return <h1>Hello, React!</h1>;
};`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">🧩 การส่งข้อมูลด้วย Props</h2>
      <pre className="p-4 mt-2 rounded-lg overflow-x-auto text-sm font-mono border">
{`const Welcome = ({ name }) => {
  return <p>สวัสดีคุณ {name}</p>;
};`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">🔁 การใช้ State ด้วย useState</h2>
      <pre className="p-4 mt-2 rounded-lg overflow-x-auto text-sm font-mono border">
{`import { useState } from "react";

const Counter = () => {
  const [count, setCount] = useState(0);
  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>เพิ่ม</button>
    </div>
  );
};`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">⏱️ React Lifecycle (เบื้องต้น)</h2>
      <p className="mt-2">
        React ใช้ <code>useEffect</code> ในการควบคุม side effects เช่น ดึงข้อมูลหรือจัดการกับ DOM:
      </p>
      <pre className="p-4 mt-2 rounded-lg overflow-x-auto text-sm font-mono border">
{`import { useEffect } from "react";

useEffect(() => {
  console.log("Component loaded!");
}, []);`}
      </pre>
    </div>
  );
};

export default ReactIntro;
