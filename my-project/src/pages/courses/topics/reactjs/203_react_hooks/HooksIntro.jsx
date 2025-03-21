import React from "react";

const HooksIntro = () => {
  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold">React Hooks คืออะไร?</h1>
      <p className="mt-4 text-lg">
        <strong>Hooks</strong> คือฟีเจอร์ที่ถูกเพิ่มเข้ามาใน React ตั้งแต่เวอร์ชัน 16.8 
        เพื่อให้สามารถใช้ state, lifecycle และ logic อื่น ๆ ได้ใน Functional Components 
        โดยไม่ต้องเขียนเป็น Class Components อีกต่อไป
      </p>

      <h2 className="text-2xl font-semibold mt-6">📌 ประโยชน์ของ Hooks</h2>
      <ul className="list-disc list-inside mt-4 space-y-2">
        <li>ลดความซับซ้อนของโค้ด ไม่ต้องใช้คลาสอีกต่อไป</li>
        <li>สามารถใช้ state และ side effects ได้ภายในฟังก์ชัน</li>
        <li>แยก logic ออกมาเป็น Custom Hooks ที่ reuse ได้</li>
        <li>เขียนและอ่านง่ายขึ้น โดยเฉพาะในโปรเจกต์ใหญ่</li>
      </ul>

      <h2 className="text-2xl font-semibold mt-6">🧩 Hooks ยอดนิยมใน React</h2>
      <ul className="list-disc list-inside mt-4 space-y-2">
        <li><code>useState</code>: จัดการค่าภายใน Component</li>
        <li><code>useEffect</code>: จัดการ side effects เช่น fetch API</li>
        <li><code>useContext</code>: ใช้ข้อมูลจาก Context API</li>
        <li><code>useRef</code>: อ้างอิง DOM หรือค่าคงที่</li>
        <li><code>useMemo / useCallback</code>: เพิ่มประสิทธิภาพการ render</li>
      </ul>

      <h2 className="text-2xl font-semibold mt-6">🎯 ตัวอย่างการใช้ useState</h2>
      <pre className="p-4 mt-4 rounded-md overflow-x-auto border bg-gray-100 dark:bg-gray-800 text-black dark:text-white text-sm">
{`import React, { useState } from "react";

const Counter = () => {
  const [count, setCount] = useState(0);
  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>เพิ่มค่า</button>
    </div>
  );
};`}
      </pre>

      <p className="mt-6">
        ตัวอย่างข้างต้นใช้ <code>useState</code> เพื่อเก็บและอัปเดตค่า <code>count</code> ซึ่งเป็นพื้นฐานสำคัญของการจัดการข้อมูลภายใน Component
      </p>
    </div>
  );
};

export default HooksIntro;
