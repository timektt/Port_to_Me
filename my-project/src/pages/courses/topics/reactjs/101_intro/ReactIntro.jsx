import React from "react";

const ReactIntro = () => {
  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold">React.js คืออะไร?</h1>
      <p className="mt-4 text-lg">
        React.js เป็น JavaScript Library ที่ใช้สำหรับพัฒนา UI ของเว็บแอปพลิเคชัน 
        โดยพัฒนาโดย Facebook และได้รับความนิยมอย่างมากในวงการ Frontend 
        เนื่องจากช่วยให้นักพัฒนาสามารถสร้าง UI ที่มีประสิทธิภาพและมีโครงสร้างที่ชัดเจน
      </p>

      <h2 className="text-2xl font-semibold mt-6">📌 คุณสมบัติหลักของ React.js</h2>
      <ul className="list-disc list-inside mt-4 space-y-2">
        <li><span className="mr-2">✅</span> <strong>Component-based:</strong> แยกโค้ดออกเป็นส่วน ๆ ที่สามารถนำกลับมาใช้ใหม่ได้</li>
        <li><span className="mr-2">✅</span> <strong>Virtual DOM:</strong> ปรับปรุงประสิทธิภาพการเรนเดอร์ UI โดยใช้โครงสร้างเสมือนของ DOM</li>
        <li><span className="mr-2">✅</span> <strong>Declarative:</strong> ใช้แนวคิดการเขียนโค้ดที่บอกว่าต้องการให้ UI แสดงผลอย่างไร</li>
        <li><span className="mr-2">✅</span> <strong>Unidirectional Data Flow:</strong> ใช้การส่งข้อมูลจากบนลงล่างผ่าน Props</li>
      </ul>

      <h2 className="text-2xl font-semibold mt-6">🎯 ตัวอย่างการสร้าง Component</h2>
      <p className="mt-4">
        React ใช้แนวคิด Component-based เราสามารถสร้าง Component ได้โดยใช้ฟังก์ชันดังนี้:
      </p>

      {/* ✅ โค้ดตัวอย่างการสร้าง React Component */}
      <pre className="p-4 mt-4 rounded-lg overflow-x-auto text-sm font-mono border">
{`const MyComponent = () => {
  return <h1>Hello, React!</h1>;
};`}
      </pre>
    </div>
  );
};

export default ReactIntro;
