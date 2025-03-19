import React from "react";

const HooksIntro = () => {
  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold">React Hooks คืออะไร?</h1>
      <p className="mt-4 text-lg">
        Hooks เป็นฟีเจอร์ที่ถูกเพิ่มเข้ามาใน React ตั้งแต่เวอร์ชัน 16.8 ช่วยให้สามารถใช้ state และ lifecycle methods ได้ใน Functional Components โดยไม่ต้องใช้ Class Components
      </p>
      
      <h2 className="text-2xl font-semibold mt-6">📌 ประโยชน์ของ Hooks</h2>
      <ul className="list-disc list-inside mt-4 space-y-2">
        <li>ลดความซับซ้อนของโค้ด ไม่ต้องใช้ Class Components</li>
        <li>สามารถใช้งาน state และ lifecycle ได้ในฟังก์ชัน</li>
        <li>แยก logic ออกเป็น reusable functions ได้ง่ายขึ้น</li>
      </ul>
      
      <h2 className="text-2xl font-semibold mt-6">🎯 ตัวอย่างการใช้ useState</h2>
      <pre className="p-4 mt-4 rounded-md overflow-x-auto border">
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
    </div>
  );
};

export default HooksIntro;
